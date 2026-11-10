/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License and the Alliance for
 * Open Media Patent License 1.0. If the BSD 3-Clause Clear License was not distributed with this
 * source code in the LICENSE file, you can obtain it at
 * aomedia.org/license/software-license/bsd-3-c-c/. If the Alliance for Open Media Patent
 * License 1.0 was not distributed with this source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include "av2_codec_config.h"
#include "frame_header_helpers.h"
#include "mux_log.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <tuple>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <av2_obu/core/bitstream_reader.h>
#include <av2_obu/core/bitstream_writer.h>
#include <av2_obu/core/temporal_unit.h>
#include <av2_obu/obus/atlas_segment_obu.h>
#include <av2_obu/obus/content_interpretation_obu.h>
#include <av2_obu/obus/layer_configuration_record_obu.h>
#include <av2_obu/obus/metadata_group_obu.h>
#include <av2_obu/obus/metadata_obu.h>
#include <av2_obu/obus/msdo_obu.h>
#include <av2_obu/obus/operating_point_set_obu.h>
#include <av2_obu/obus/sequence_header_obu.h>

namespace av2_obu {

namespace {

// Decode an Annex B LEB128 size field from [data, data + size).
// Returns number of bytes consumed by LEB128, or 0 on error.
size_t read_leb128(const uint8_t* data, size_t size, uint64_t& out_value) {
  out_value = 0;
  uint32_t shift = 0;
  for (size_t i = 0; i < size && i < 8; ++i) {
    uint8_t b = data[i];
    out_value |= static_cast<uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) {
      return i + 1;
    }
    shift += 7;
  }
  return 0;
}

// Parse OBU header fields (type, xlayer_id, mlayer_id) from an Annex B framed OBU.
bool parse_annexb_obu_header(const uint8_t* data, size_t size, OBUType& out_type,
                             uint8_t& out_xlayer_id, uint8_t& out_mlayer_id,
                             size_t& out_total_len) {
  uint64_t obu_len = 0;
  size_t leb_len = read_leb128(data, size, obu_len);
  if (leb_len == 0 || obu_len == 0 || leb_len + obu_len > size) {
    return false;
  }
  const uint8_t* hdr = data + leb_len;
  uint8_t ext_flag = (hdr[0] >> 7) & 0x01;
  uint32_t type_val = (hdr[0] >> 2) & 0x1F;
  out_type = to_obu_type(type_val);
  if (ext_flag) {
    if (obu_len < 2) return false;
    out_mlayer_id = (hdr[1] >> 5) & 0x07;
    out_xlayer_id = hdr[1] & 0x1F;
  } else {
    out_mlayer_id = 0;
    // Per AV2 spec §6.2.2: when obu_extension_flag == 0, obu_xlayer_id is inferred
    // as GLOBAL_XLAYER_ID (31) for TD, MSDO, LCR, OPS, ATLAS_SEGMENT, and PADDING,
    // and 0 for other OBUs.
    if (out_type == OBUType::TEMPORAL_DELIMITER || out_type == OBUType::MSDO ||
        out_type == OBUType::LAYER_CONFIGURATION_RECORD ||
        out_type == OBUType::OPERATING_POINT_SET || out_type == OBUType::ATLAS_SEGMENT ||
        out_type == OBUType::PADDING) {
      out_xlayer_id = GLOBAL_XLAYER_ID;
    } else {
      out_xlayer_id = 0;
    }
  }
  out_total_len = leb_len + static_cast<size_t>(obu_len);
  return true;
}

// Canonical ordering key for config OBUs in a Temporal Unit (AV2 §7.3.5):
//   Group 0 (global scope, xlayer_id == 31):
//     0: OBU_MSDO
//     1: OBU_LAYER_CONFIGURATION_RECORD
//     2: OBU_OPERATING_POINT_SET
//     3: OBU_ATLAS_SEGMENT
//     4: OBU_METADATA_SHORT / OBU_METADATA_GROUP (prefix)
//   Group 1 (grouped high-level syntax block, xlayer_id < 31, ascending xlayer_id):
//     0: OBU_LAYER_CONFIGURATION_RECORD
//     1: OBU_OPERATING_POINT_SET
//     2: OBU_ATLAS_SEGMENT
//     3: OBU_SEQUENCE_HEADER
std::tuple<int, int, int> config_obu_sort_key(
    const OutOfBandAV2CodecConfigurationBox::ConfigEntry& e) {
  if (e.xlayer_id == GLOBAL_XLAYER_ID) {
    int type_rank = 99;
    switch (e.obu_type) {
      case OBUType::MSDO:
        type_rank = 0;
        break;
      case OBUType::LAYER_CONFIGURATION_RECORD:
        type_rank = 1;
        break;
      case OBUType::OPERATING_POINT_SET:
        type_rank = 2;
        break;
      case OBUType::ATLAS_SEGMENT:
        type_rank = 3;
        break;
      case OBUType::METADATA:
      case OBUType::METADATA_GROUP:
        type_rank = 4;
        break;
      default:
        type_rank = 5;
        break;
    }
    return {0, 0, type_rank};
  }
  int type_rank = 99;
  switch (e.obu_type) {
    case OBUType::LAYER_CONFIGURATION_RECORD:
      type_rank = 0;
      break;
    case OBUType::OPERATING_POINT_SET:
      type_rank = 1;
      break;
    case OBUType::ATLAS_SEGMENT:
      type_rank = 2;
      break;
    case OBUType::SEQUENCE_HEADER:
      type_rank = 3;
      break;
    default:
      type_rank = 4;
      break;
  }
  return {1, static_cast<int>(e.xlayer_id), type_rank};
}

}  // namespace

bool OutOfBandAV2CodecConfigurationBox::append_config_obu(const BaseOBU& obu) {
  if (!input_file_) {
    MUX_ERROR("append_config_obu: no input file bound");
    return false;
  }
  if (!is_config_obu_instance(obu)) {
    MUX_ERROR("append_config_obu: OBU type {} is not allowed in av2C",
              static_cast<int>(obu.type()));
    return false;
  }

  const auto& pos = obu.position();
  size_t total = pos.size_field_len + pos.header_len + pos.payload_size;

  auto saved_pos = input_file_->tellg();

  input_file_->clear();
  input_file_->seekg(pos.start_pos);
  if (!*input_file_) {
    MUX_ERROR("Failed to seek to OBU at offset {}", static_cast<long long>(pos.start_pos));
    input_file_->clear();
    input_file_->seekg(saved_pos);
    return false;
  }

  ConfigEntry entry;
  entry.obu = &obu;
  entry.obu_type = obu.type();
  entry.xlayer_id = obu.header().get_xlayer_id();
  entry.mlayer_id = obu.header().get_mlayer_id();
  entry.bytes.resize(total);
  if (!input_file_->read(reinterpret_cast<char*>(entry.bytes.data()),
                         static_cast<std::streamsize>(total))) {
    MUX_ERROR("Failed to read {} bytes for OBU at offset {}", total,
              static_cast<long long>(pos.start_pos));
    input_file_->clear();
    input_file_->seekg(saved_pos);
    return false;
  }

  // Deduplicate bit-identical config OBUs within the same av2C (e.g., when a TU has an OBU
  // both in the grouped HLS block and repeated inside the coded extended layer unit).
  for (const auto& existing : config_obus) {
    if (existing.bytes == entry.bytes) {
      input_file_->clear();
      input_file_->seekg(saved_pos);
      return true;
    }
  }

  config_obus.push_back(std::move(entry));
  input_file_->clear();
  input_file_->seekg(saved_pos);
  return true;
}

bool OutOfBandAV2CodecConfigurationBox::append_raw_config_obu(std::vector<uint8_t> bytes) {
  OBUType obu_type = OBUType::UNKNOWN;
  uint8_t xlayer_id = GLOBAL_XLAYER_ID;
  uint8_t mlayer_id = 0;
  size_t total_len = 0;
  if (!parse_annexb_obu_header(bytes.data(), bytes.size(), obu_type, xlayer_id, mlayer_id,
                               total_len) ||
      total_len != bytes.size()) {
    MUX_ERROR("append_raw_config_obu: invalid Annex B framing");
    return false;
  }
  if (is_forbidden_in_isobmff(obu_type)) {
    MUX_ERROR("append_raw_config_obu: OBU type {} is forbidden in av2C",
              static_cast<int>(obu_type));
    return false;
  }
  if (!is_config_obu(obu_type) &&
      !((obu_type == OBUType::METADATA || obu_type == OBUType::METADATA_GROUP) &&
        xlayer_id == GLOBAL_XLAYER_ID)) {
    MUX_ERROR("append_raw_config_obu: OBU type {} is not allowed in av2C",
              static_cast<int>(obu_type));
    return false;
  }

  for (const auto& existing : config_obus) {
    if (existing.bytes == bytes) {
      return true;
    }
  }

  ConfigEntry entry;
  entry.obu = nullptr;
  entry.obu_type = obu_type;
  entry.xlayer_id = xlayer_id;
  entry.mlayer_id = mlayer_id;
  entry.bytes = std::move(bytes);
  config_obus.push_back(std::move(entry));
  return true;
}

void OutOfBandAV2CodecConfigurationBox::sort_in_tu_order() {
  std::stable_sort(config_obus.begin(), config_obus.end(),
                   [](const ConfigEntry& a, const ConfigEntry& b) {
                     return config_obu_sort_key(a) < config_obu_sort_key(b);
                   });
}

std::vector<uint8_t> OutOfBandAV2CodecConfigurationBox::concatenated_config_obus() const {
  std::vector<uint8_t> out;
  for (const auto& entry : config_obus) {
    out.insert(out.end(), entry.bytes.begin(), entry.bytes.end());
  }
  return out;
}

std::vector<uint8_t> OutOfBandAV2CodecConfigurationBox::serialize() const {
  if (config_obus.empty() || config_obus.size() > 256) {
    MUX_ERROR("av2C requires between 1 and 256 config OBUs (have {})", config_obus.size());
    return {};
  }

  std::vector<uint8_t> out;
  out.push_back(reserved1);
  out.push_back(static_cast<uint8_t>(config_obus.size() - 1));
  for (const auto& entry : config_obus) {
    out.insert(out.end(), entry.bytes.begin(), entry.bytes.end());
  }
  MUX_DEBUG("Serialized av2C: {} bytes (2-byte header + {} config_obu(s))", out.size(),
            config_obus.size());
  return out;
}

bool OutOfBandAV2CodecConfigurationBox::parse(const uint8_t* data, size_t size,
                                              OutOfBandAV2CodecConfigurationBox& out) {
  if (!data || size < 2) {
    return false;
  }
  out = OutOfBandAV2CodecConfigurationBox{};
  out.reserved1 = data[0];  // Readers SHALL ignore its value
  const size_t expected_count = static_cast<size_t>(data[1]) + 1;

  size_t offset = 2;
  for (size_t i = 0; i < expected_count; ++i) {
    if (offset >= size) {
      return false;
    }
    OBUType obu_type = OBUType::UNKNOWN;
    uint8_t xlayer_id = GLOBAL_XLAYER_ID;
    uint8_t mlayer_id = 0;
    size_t obu_total = 0;
    if (!parse_annexb_obu_header(data + offset, size - offset, obu_type, xlayer_id, mlayer_id,
                                 obu_total)) {
      return false;
    }
    if (is_forbidden_in_isobmff(obu_type)) {
      return false;
    }
    if (!is_config_obu(obu_type)) {
      if ((obu_type != OBUType::METADATA && obu_type != OBUType::METADATA_GROUP) ||
          xlayer_id != GLOBAL_XLAYER_ID) {
        return false;
      }
      // Verify metadata_is_suffix == 0 (first bit of metadata OBU payload).
      uint64_t obu_len = 0;
      size_t leb_len = read_leb128(data + offset, size - offset, obu_len);
      uint8_t ext_flag = (data[offset + leb_len] >> 7) & 0x01;
      size_t hdr_len = ext_flag ? 2 : 1;
      if (obu_len <= hdr_len) return false;
      uint8_t first_payload_byte = data[offset + leb_len + hdr_len];
      uint8_t metadata_is_suffix = (first_payload_byte >> 7) & 0x01;
      if (metadata_is_suffix != 0) return false;
    }

    ConfigEntry entry;
    entry.obu = nullptr;
    entry.obu_type = obu_type;
    entry.xlayer_id = xlayer_id;
    entry.mlayer_id = mlayer_id;
    entry.bytes.assign(data + offset, data + offset + obu_total);
    out.config_obus.push_back(std::move(entry));
    offset += obu_total;
  }

  // Unrecognized trailing bytes after config_obus_count_minus1 + 1 OBUs are ignored per ISOBMFF.

  // Verify TU ordering and presence of the grouped high-level syntax block (xlayer_id < 31).
  bool has_grouped_hls = false;
  for (size_t i = 0; i < out.config_obus.size(); ++i) {
    if (out.config_obus[i].xlayer_id != GLOBAL_XLAYER_ID) {
      has_grouped_hls = true;
    }
    if (i > 0 &&
        config_obu_sort_key(out.config_obus[i]) < config_obu_sort_key(out.config_obus[i - 1])) {
      return false;
    }
  }
  if (!has_grouped_hls) {
    return false;
  }
  return true;
}

// ==================== InBandAV2CodecConfigurationBox ('av2c') ====================

namespace {

uint32_t bit_depth_from_idc(uint8_t idc) {
  if (idc == 1) return 8;
  if (idc == 0) return 10;
  if (idc == 2) return 12;
  return 12u + idc;
}

}  // namespace

uint8_t combine_av2_profiles(uint8_t a, uint8_t b) {
  a &= 0x1F;
  b &= 0x1F;
  if (a == b) return a;
  const uint8_t lo = std::min(a, b);
  const uint8_t hi = std::max(a, b);
  if (lo <= 1) return hi;
  // Profiles 2 (Main_420_10_IP2), 3 (Main_422_10_IP1), and 4 (Main_444_10_IP1) are incomparable:
  // profile 2 allows IP 2, profile 3 allows 4:2:2, and profile 4 allows 4:4:4. Any pair of
  // distinct profiles among {2, 3, 4} requires profile 5 (Main_444C_12_IP2).
  if (hi <= 4) return 5;
  return hi;
}

std::vector<uint8_t> InBandAV2CodecConfigurationBox::serialize() const {
  uint8_t b0 = static_cast<uint8_t>(((max_num_streams_minus1 & 0x03) << 6) |
                                    ((max_profile_idc & 0x1F) << 1) |
                                    ((max_level_idx >> 4) & 0x01));
  uint8_t b1 = static_cast<uint8_t>(((max_level_idx & 0x0F) << 4) |
                                    ((max_tier & 0x01) << 3) |
                                    (max_bit_depth_idc & 0x07));
  return {b0, b1};
}

bool InBandAV2CodecConfigurationBox::parse(const uint8_t* data, size_t size,
                                           InBandAV2CodecConfigurationBox& out) {
  if (!data || size < 2) return false;
  out.max_num_streams_minus1 = (data[0] >> 6) & 0x03;
  out.max_profile_idc = (data[0] >> 1) & 0x1F;
  out.max_level_idx = static_cast<uint8_t>(((data[0] & 0x01) << 4) | ((data[1] >> 4) & 0x0F));
  out.max_tier = (data[1] >> 3) & 0x01;
  out.max_bit_depth_idc = data[1] & 0x07;
  return true;
}

InBandAV2CodecConfigurationBox InBandAV2CodecConfigurationBox::from_parser(
    const OBUParser& parser) {
  InBandAV2CodecConfigurationBox box;
  bool has_profile = false;
  bool has_bit_depth = false;
  auto update_bit_depth_idc = [&](uint8_t idc) {
    idc &= 0x07;
    if (!has_bit_depth ||
        bit_depth_from_idc(idc) > bit_depth_from_idc(box.max_bit_depth_idc)) {
      box.max_bit_depth_idc = idc;
      has_bit_depth = true;
    }
  };

  for (const auto& obu : parser.obus()) {
    if (obu->type() == OBUType::SEQUENCE_HEADER) {
      const auto* sh_obu = dynamic_cast<const SequenceHeaderOBU*>(obu.get());
      if (!sh_obu) continue;
      const auto& sh = sh_obu->sequence_header();
      const uint8_t p = static_cast<uint8_t>(sh.seq_profile_idc);
      box.max_profile_idc = has_profile ? combine_av2_profiles(box.max_profile_idc, p) : p;
      has_profile = true;
      box.max_level_idx =
          std::max<uint8_t>(box.max_level_idx, static_cast<uint8_t>(sh.seq_level_idx));
      box.max_tier = std::max<uint8_t>(box.max_tier, static_cast<uint8_t>(sh.seq_tier));
      update_bit_depth_idc(static_cast<uint8_t>(sh.bit_depth_idc));
    } else if (obu->type() == OBUType::MSDO) {
      const auto* msdo = dynamic_cast<const MSDOOBU*>(obu.get());
      if (!msdo) continue;
      if (msdo->num_streams() >= 2) {
        uint8_t minus1 = static_cast<uint8_t>(std::min<uint32_t>(3, msdo->num_streams() - 1));
        box.max_num_streams_minus1 = std::max(box.max_num_streams_minus1, minus1);
      }
      const uint8_t p = static_cast<uint8_t>(msdo->multistream_profile_idc());
      box.max_profile_idc = has_profile ? combine_av2_profiles(box.max_profile_idc, p) : p;
      has_profile = true;
      box.max_level_idx = std::max<uint8_t>(box.max_level_idx,
                                            static_cast<uint8_t>(msdo->multistream_level_idx()));
      box.max_tier =
          std::max<uint8_t>(box.max_tier, static_cast<uint8_t>(msdo->multistream_tier()));
    } else if (obu->type() == OBUType::LAYER_CONFIGURATION_RECORD) {
      const auto* lcr = dynamic_cast<const LayerConfigurationRecordOBU*>(obu.get());
      if (!lcr) continue;
      if (lcr->is_global()) {
        for (const auto& gp : lcr->global_payloads()) {
          if (gp.xlayer_info.rep_info_present && gp.xlayer_info.rep_info.format_info_present) {
            update_bit_depth_idc(static_cast<uint8_t>(gp.xlayer_info.rep_info.bit_depth_idc));
          }
        }
      } else {
        const auto& xinfo = lcr->local_xlayer_info();
        if (xinfo.rep_info_present && xinfo.rep_info.format_info_present) {
          update_bit_depth_idc(static_cast<uint8_t>(xinfo.rep_info.bit_depth_idc));
        }
      }
    }
  }
  return box;
}

// ==================== PixiBox ('pixi') ====================

std::vector<uint8_t> PixiBox::serialize_payload() const {
  if (channels.empty() || channels.size() > 255) {
    MUX_ERROR("pixi requires between 1 and 255 channels (have {})", channels.size());
    return {};
  }

  std::vector<uint8_t> out;
  // FullBox version (8 bits) + flags (24 bits)
  out.push_back(version);
  out.push_back(static_cast<uint8_t>((flags >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((flags >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>(flags & 0xFF));

  out.push_back(static_cast<uint8_t>(channels.size()));
  for (const auto& ch : channels) {
    out.push_back(ch.bits_per_channel);
  }

  if ((flags & 1u) != 0) {
    for (const auto& ch : channels) {
      const bool label_flag = !ch.channel_label.empty();
      uint8_t b = static_cast<uint8_t>(((ch.channel_idc & 0x07) << 5) |
                                       ((ch.component_format & 0x03) << 2) |
                                       ((ch.subsampling_flag ? 1 : 0) << 1) |
                                       (label_flag ? 1 : 0));
      out.push_back(b);
      if (ch.subsampling_flag) {
        uint8_t sub = static_cast<uint8_t>(((ch.subsampling_type & 0x0F) << 4) |
                                           (ch.subsampling_location & 0x0F));
        out.push_back(sub);
      }
      if (label_flag) {
        out.insert(out.end(), ch.channel_label.begin(), ch.channel_label.end());
        out.push_back(0);  // Null terminator for utf8string
      }
    }
  }
  return out;
}

bool PixiBox::parse_payload(const uint8_t* data, size_t size, PixiBox& out) {
  if (!data || size < 5) return false;
  out = PixiBox{};
  out.version = data[0];
  out.flags = (static_cast<uint32_t>(data[1]) << 16) | (static_cast<uint32_t>(data[2]) << 8) |
              static_cast<uint32_t>(data[3]);
  if (out.version != 0) return false;

  uint8_t num_channels = data[4];
  if (num_channels == 0 || size < 5u + num_channels) return false;

  out.channels.resize(num_channels);
  size_t offset = 5;
  for (uint8_t i = 0; i < num_channels; ++i) {
    if (data[offset] == 0) return false;  // bits_per_channel shall not be 0
    out.channels[i].bits_per_channel = data[offset++];
  }

  if ((out.flags & 1u) != 0) {
    for (uint8_t i = 0; i < num_channels; ++i) {
      if (offset >= size) return false;
      uint8_t b = data[offset++];
      auto& ch = out.channels[i];
      ch.channel_idc = (b >> 5) & 0x07;
      ch.component_format = (b >> 2) & 0x03;
      ch.subsampling_flag = ((b >> 1) & 0x01) != 0;
      bool label_flag = (b & 0x01) != 0;
      if (ch.subsampling_flag) {
        if (offset >= size) return false;
        uint8_t sub = data[offset++];
        ch.subsampling_type = (sub >> 4) & 0x0F;
        ch.subsampling_location = sub & 0x0F;
      }
      if (label_flag) {
        size_t start = offset;
        while (offset < size && data[offset] != 0) {
          ++offset;
        }
        if (offset >= size) return false;  // Missing null terminator
        ch.channel_label.assign(reinterpret_cast<const char*>(data + start), offset - start);
        ++offset;  // Skip null terminator
      }
    }
  }

  // Unrecognized trailing bytes in the box are ignored per ISOBMFF.
  return offset <= size;
}

PixiBox PixiBox::from_parser(const OBUParser& parser, uint8_t target_xlayer_id) {
  PixiBox pixi;
  pixi.version = 0;
  pixi.flags = 1;

  // Map xlayer_id -> active SequenceHeader
  std::map<uint8_t, const AV2SequenceHeader*> sh_by_xlayer;
  // Map xlayer_id -> ContentInterpretationOBU
  std::map<uint8_t, const ContentInterpretationOBU*> ci_by_xlayer;
  // Set of active mlayer_id values within target_xlayer_id (iterated in increasing mlayer_id order
  // per av2-isobmff #channels).
  std::set<uint8_t> active_mlayers;

  // Map mlayer_id (within target_xlayer_id) -> (layer_type, auxiliary_type)
  std::map<uint8_t, std::pair<uint32_t, uint32_t>> layer_roles;

  for (const auto& obu : parser.obus()) {
    const uint8_t xid = obu->header().get_xlayer_id();
    const uint8_t mid = obu->header().get_mlayer_id();
    if (obu->type() == OBUType::SEQUENCE_HEADER) {
      if (const auto* sh = dynamic_cast<const SequenceHeaderOBU*>(obu.get())) {
        if (sh_by_xlayer.find(xid) == sh_by_xlayer.end()) {
          sh_by_xlayer[xid] = &sh->sequence_header();
        }
        if (xid == target_xlayer_id) {
          active_mlayers.insert(0);
        }
      }
    } else if (obu->type() == OBUType::CONTENT_INTERPRETATION) {
      if (const auto* ci = dynamic_cast<const ContentInterpretationOBU*>(obu.get())) {
        if (ci_by_xlayer.find(xid) == ci_by_xlayer.end()) {
          ci_by_xlayer[xid] = ci;
        }
      }
    } else if (obu->type() == OBUType::LAYER_CONFIGURATION_RECORD) {
      if (const auto* lcr = dynamic_cast<const LayerConfigurationRecordOBU*>(obu.get())) {
        if (lcr->is_global()) {
          for (const auto& gp : lcr->global_payloads()) {
            const uint8_t gp_xid = static_cast<uint8_t>(gp.xlayer_id);
            if (gp_xid != target_xlayer_id) continue;
            if (gp.xlayer_info.embedded_layer_info_present) {
              for (const auto& el : gp.xlayer_info.embedded_layers) {
                const uint8_t el_mid = static_cast<uint8_t>(el.mlayer_id);
                active_mlayers.insert(el_mid);
                layer_roles[el_mid] = {el.layer_type, el.auxiliary_type};
              }
            } else {
              active_mlayers.insert(0);
            }
          }
        } else if (xid == target_xlayer_id) {
          const auto& xinfo = lcr->local_xlayer_info();
          if (xinfo.embedded_layer_info_present) {
            for (const auto& el : xinfo.embedded_layers) {
              const uint8_t el_mid = static_cast<uint8_t>(el.mlayer_id);
              active_mlayers.insert(el_mid);
              layer_roles[el_mid] = {el.layer_type, el.auxiliary_type};
            }
          } else {
            active_mlayers.insert(0);
          }
        }
      }
    } else if (obu->type() == OBUType::ATLAS_SEGMENT) {
      if (const auto* atlas = dynamic_cast<const AtlasSegmentOBU*>(obu.get())) {
        if (atlas->mode_idc() == AtlasSegmentOBU::ModeIdc::MULTISTREAM_ALPHA_ATLAS &&
            atlas->msi_alpha_segments_present()) {
          for (const auto& seg : atlas->segments()) {
            if (seg.alpha_segment_flag &&
                static_cast<uint8_t>(seg.input_stream_id) == target_xlayer_id) {
              if (layer_roles.find(0) == layer_roles.end()) {
                layer_roles[0] = {AUX_LAYER, ALPHA_AUX};
              }
            }
          }
        }
      }
    } else if (frame_header_of(obu.get()) != nullptr && xid == target_xlayer_id) {
      active_mlayers.insert(mid);
    }
  }

  const AV2SequenceHeader* sh = nullptr;
  if (auto it = sh_by_xlayer.find(target_xlayer_id); it != sh_by_xlayer.end()) {
    sh = it->second;
  } else if (target_xlayer_id == 0 && !sh_by_xlayer.empty()) {
    sh = sh_by_xlayer.begin()->second;
  }
  if (!sh) {
    return pixi;
  }
  if (active_mlayers.empty()) {
    active_mlayers.insert(0);
  }

  // Track used colour channel_idc values for monochrome multi-layer (e.g., CMYK: 2, 3, 4, 7).
  const uint8_t kMonoColourIdcs[4] = {2, 3, 4, 7};
  size_t mono_colour_idx = 0;

  const uint8_t bit_depth = static_cast<uint8_t>(sh->BitDepth);
  const uint64_t cf = sh->chroma_format_idc;  // 0=4:2:0, 1=4:0:0, 2=4:4:4, 3=4:2:2

  uint8_t sub_type = 2;  // 4:2:0 (CHROMA_FORMAT_420)
  if (cf == CHROMA_FORMAT_444) {
    sub_type = 0;  // 4:4:4
  } else if (cf == CHROMA_FORMAT_422) {
    sub_type = 1;  // 4:2:2
  }

  uint8_t sub_loc = 0;
  if (auto ci_it = ci_by_xlayer.find(target_xlayer_id); ci_it != ci_by_xlayer.end()) {
    const ContentInterpretationOBU* ci = ci_it->second;
    if (ci->has_chroma_sample_position() && ci->chroma_sample_position_top() <= 5) {
      sub_loc = static_cast<uint8_t>(ci->chroma_sample_position_top());
    }
  }

  for (const uint8_t mid : active_mlayers) {
    uint32_t layer_type = TEXTURE_LAYER;
    uint32_t aux_type = 0;
    if (auto r_it = layer_roles.find(mid); r_it != layer_roles.end()) {
      layer_type = r_it->second.first;
      aux_type = r_it->second.second;
    }

    if (layer_type == TEXTURE_LAYER) {
      if (cf == 1) {  // CHROMA_FORMAT_400 (monochrome, 1 plane)
        PixiChannel c;
        c.bits_per_channel = bit_depth;
        c.channel_idc =
            mono_colour_idx < 4 ? kMonoColourIdcs[mono_colour_idx++] : static_cast<uint8_t>(1);
        c.component_format = 0;
        c.subsampling_flag = false;
        pixi.channels.push_back(c);
      } else {  // 3 planes (Y/R, U/G, V/B)
        PixiChannel c0;
        c0.bits_per_channel = bit_depth;
        c0.channel_idc = 2;
        c0.component_format = 0;
        c0.subsampling_flag = false;
        pixi.channels.push_back(c0);

        for (uint8_t ch_idc : {uint8_t{3}, uint8_t{4}}) {
          PixiChannel c;
          c.bits_per_channel = bit_depth;
          c.channel_idc = ch_idc;
          c.component_format = 0;
          c.subsampling_flag = true;
          c.subsampling_type = sub_type;
          c.subsampling_location = sub_loc;
          pixi.channels.push_back(c);
        }
      }
    } else {  // AUX_LAYER
      if (aux_type == ALPHA_AUX || aux_type == DEPTH_AUX) {
        const uint8_t target_idc = (aux_type == ALPHA_AUX) ? 5 : 6;
        PixiChannel c0;
        c0.bits_per_channel = bit_depth;
        c0.channel_idc = target_idc;
        c0.component_format = 0;
        c0.subsampling_flag = false;
        pixi.channels.push_back(c0);

        if (cf != 1) {
          // Chroma planes in a 3-plane embedded layer carrying alpha/depth are Unused (0).
          for (int p = 1; p < 3; ++p) {
            PixiChannel unused_ch;
            unused_ch.bits_per_channel = bit_depth;
            unused_ch.channel_idc = 0;  // Unused
            unused_ch.component_format = 0;
            unused_ch.subsampling_flag = false;
            pixi.channels.push_back(unused_ch);
          }
        }
      } else {
        // Segmentation, gain map, or other auxiliary layer -> Unspecified (1)
        const int planes = (cf == 1) ? 1 : 3;
        for (int p = 0; p < planes; ++p) {
          PixiChannel unspec_ch;
          unspec_ch.bits_per_channel = bit_depth;
          unspec_ch.channel_idc = 1;  // Unspecified
          unspec_ch.component_format = 0;
          unspec_ch.subsampling_flag = false;
          pixi.channels.push_back(unspec_ch);
        }
      }
    }
  }

  return pixi;
}

static char to_base32_char(uint8_t val) {
  val &= 0x1F;
  return val < 10 ? static_cast<char>('0' + val) : static_cast<char>('A' + (val - 10));
}

std::string build_codecs_string(const std::string& sample_entry_4cc,
                                uint8_t max_profile_idc,
                                uint8_t max_level_idx,
                                uint8_t max_tier,
                                uint8_t max_bit_depth_idc) {
  const char tier_char = (max_tier == 0) ? 'M' : 'H';
  return fmt::format("{}.{}{}{}{}", sample_entry_4cc, to_base32_char(max_profile_idc),
                     to_base32_char(max_level_idx), tier_char,
                     static_cast<unsigned>(max_bit_depth_idc & 0x07));
}

std::string build_codecs_string(const std::string& sample_entry_4cc,
                                const InBandAV2CodecConfigurationBox& inband) {
  return build_codecs_string(sample_entry_4cc, inband.max_profile_idc, inband.max_level_idx,
                             inband.max_tier, inband.max_bit_depth_idc);
}

std::string build_codecs_string(const std::string& sample_entry_4cc,
                                const AV2SequenceHeader& first_sh,
                                const PixiBox& /*pixi*/,
                                const ColrInfo& /*colr*/) {
  return build_codecs_string(sample_entry_4cc, static_cast<uint8_t>(first_sh.seq_profile_idc),
                             static_cast<uint8_t>(first_sh.seq_level_idx),
                             static_cast<uint8_t>(first_sh.seq_tier),
                             static_cast<uint8_t>(first_sh.bit_depth_idc));
}

std::vector<std::pair<uint8_t, uint8_t>> collect_ordered_layers(const OBUParser& parser) {
  std::set<std::pair<uint8_t, uint8_t>> layers;
  for (const auto& obu : parser.obus()) {
    const uint8_t xid = obu->header().get_xlayer_id();
    const uint8_t mid = obu->header().get_mlayer_id();
    if (obu->type() == OBUType::SEQUENCE_HEADER && xid != GLOBAL_XLAYER_ID) {
      layers.insert({xid, 0});
    } else if (obu->type() == OBUType::LAYER_CONFIGURATION_RECORD) {
      if (const auto* lcr = dynamic_cast<const LayerConfigurationRecordOBU*>(obu.get())) {
        if (lcr->is_global()) {
          for (const auto& gp : lcr->global_payloads()) {
            const uint8_t gp_xid = static_cast<uint8_t>(gp.xlayer_id);
            if (gp.xlayer_info.embedded_layer_info_present) {
              for (const auto& el : gp.xlayer_info.embedded_layers) {
                layers.insert({gp_xid, static_cast<uint8_t>(el.mlayer_id)});
              }
            } else {
              layers.insert({gp_xid, 0});
            }
          }
        } else if (xid != GLOBAL_XLAYER_ID) {
          const auto& xinfo = lcr->local_xlayer_info();
          if (xinfo.embedded_layer_info_present) {
            for (const auto& el : xinfo.embedded_layers) {
              layers.insert({xid, static_cast<uint8_t>(el.mlayer_id)});
            }
          } else {
            layers.insert({xid, 0});
          }
        }
      }
    } else if (frame_header_of(obu.get()) != nullptr && xid != GLOBAL_XLAYER_ID) {
      layers.insert({xid, mid});
    }
  }
  return {layers.begin(), layers.end()};
}

int find_layer_order_index(const std::vector<std::pair<uint8_t, uint8_t>>& ordered_layers,
                           uint8_t xlayer_id, uint8_t mlayer_id) {
  for (size_t i = 0; i < ordered_layers.size(); ++i) {
    if (ordered_layers[i].first == xlayer_id && ordered_layers[i].second == mlayer_id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

uint64_t compute_layer_bitset(const std::vector<std::pair<uint8_t, uint8_t>>& ordered_layers,
                              const std::vector<std::pair<uint8_t, uint8_t>>& selected_layers) {
  uint64_t mask = 0;
  for (const auto& [xid, mid] : selected_layers) {
    int idx = find_layer_order_index(ordered_layers, xid, mid);
    if (idx >= 0 && idx < 64) {
      mask |= (1ULL << idx);
    }
  }
  return mask;
}

bool make_reconstruction_point_id(uint8_t obu_xlayer_id, uint8_t ops_id, uint8_t op_index,
                                  uint16_t& out_id) {
  if (obu_xlayer_id > 31 || ops_id > 15 || op_index >= 7) {
    return false;
  }
  out_id = static_cast<uint16_t>(((obu_xlayer_id & 0x1F) << 7) | ((ops_id & 0x0F) << 3) |
                                 (op_index & 0x07));
  return true;
}

bool parse_reconstruction_point_id(uint16_t id, uint8_t& out_obu_xlayer_id, uint8_t& out_ops_id,
                                   uint8_t& out_op_index) {
  if (id > 0x0FFF) return false;
  uint8_t op_idx = static_cast<uint8_t>(id & 0x07);
  if (op_idx == 7) return false;  // 3 LSBs shall not be all ones (ops_cnt <= 7 -> op_index in 0..6)
  out_obu_xlayer_id = static_cast<uint8_t>((id >> 7) & 0x1F);
  out_ops_id = static_cast<uint8_t>((id >> 3) & 0x0F);
  out_op_index = op_idx;
  return true;
}

}  // namespace av2_obu
