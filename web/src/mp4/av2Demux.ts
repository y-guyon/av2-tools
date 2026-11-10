// Demux an AV2-in-ISOBMFF (.mp4) file into an AV2 elementary bitstream (.obu)
// in the browser, so the existing OBU parser can process it unchanged.
//
// Mirrors the C++ av2_demux (av2-tools/apps/av2_demux/av2_demuxer.cpp):
//   - read the av2C/av2c configOBUs from every stsd sample entry (lines 244-296),
//   - for every sample emit a Temporal Delimiter, splice that sample's desc_idx
//     configOBUs at sample 0, sample-entry changes, and random access points,
//     discard Possibly out-of-band OBUs in 'av2E' samples, then emit remaining
//     OBUs (lines 323-389).
//
// mp4box.js handles ISOBMFF parsing + sample extraction. It does not know AV2,
// and mp4box 2.4.1 dropped the classic createBoxCtor registration API, so the
// one box it can't surface - av2C - is read here with a small manual box walk.

import { createFile } from 'mp4box';

// OutOfBandAV2CodecConfigurationBox ('av2C') payload layout (per av2-isobmff):
// a 2-byte prefix (reserved1, config_obus_count_minus1) followed by config_obu[].
const AV2C_PREFIX_LEN = 2;
const AV2_SAMPLE_ENTRY_TYPES = new Set(['av2E', 'av2e']);
// VisualSampleEntry fixed fields preceding the child boxes (matches the C++
// kVisualSampleEntryFixedSize in av2_demuxer.cpp line 143): 8-byte box header + 78 bytes.
const SAMPLE_ENTRY_HEADER_LEN = 8 + 78;
// Annex-B framed Temporal Delimiter OBU (leb128 size = 1, obu_type = 2).
const TD_BYTES = Uint8Array.of(0x01, 0x08);

function u32(d: Uint8Array, o: number): number {
  return (d[o] * 0x1000000 + (d[o + 1] << 16) + (d[o + 2] << 8) + d[o + 3]) >>> 0;
}

function fourCC(d: Uint8Array, o: number): string {
  return String.fromCharCode(d[o], d[o + 1], d[o + 2], d[o + 3]);
}

// First child box of `type` within [start, end). Returns [boxStart, boxEnd] or null.
function findChild(
  d: Uint8Array,
  start: number,
  end: number,
  type: string,
): [number, number] | null {
  let o = start;
  while (o + 8 <= end) {
    const size = u32(d, o);
    const boxEnd = size === 0 ? end : o + size;
    if (size !== 0 && (size < 8 || boxEnd > end)) break;
    if (fourCC(d, o + 4) === type) return [o, boxEnd];
    o = boxEnd;
  }
  return null;
}

// Walk one trak's mdia > minf > stbl > stsd > av2E/av2e > av2C/av2c; return the
// per-sample-entry configOBUs (av2C payload minus the 2-byte prefix, or empty Uint8Array
// for in-band av2c), or null if not an AV2 track.
interface SampleEntryConfig {
  isOob: boolean;
  configObus: Uint8Array;
}

function configObusFromTrak(d: Uint8Array, start: number, end: number): SampleEntryConfig[] | null {
  let node = findChild(d, start, end, 'mdia');
  if (!node) return null;
  node = findChild(d, node[0] + 8, node[1], 'minf');
  if (!node) return null;
  node = findChild(d, node[0] + 8, node[1], 'stbl');
  if (!node) return null;
  const stsd = findChild(d, node[0] + 8, node[1], 'stsd');
  if (!stsd || stsd[0] + 16 > stsd[1]) return null;
  // stsd is a FullBox: 8-byte header + 4 (version/flags) + 4 (entry_count).
  const entryCount = u32(d, stsd[0] + 12);
  let seStart = stsd[0] + 16;
  const entries: SampleEntryConfig[] = [];
  for (let e = 0; e < entryCount && seStart + 8 <= stsd[1]; e++) {
    const seSize = u32(d, seStart);
    const seEnd = seSize === 0 ? stsd[1] : Math.min(seStart + seSize, stsd[1]);
    if (seSize !== 0 && seSize < SAMPLE_ENTRY_HEADER_LEN) return null;
    const seType = fourCC(d, seStart + 4);
    if (!AV2_SAMPLE_ENTRY_TYPES.has(seType)) return null;
    if (seType === 'av2e') {
      const av2cInband = findChild(d, seStart + SAMPLE_ENTRY_HEADER_LEN, seEnd, 'av2c');
      if (!av2cInband || av2cInband[1] - (av2cInband[0] + 8) < 2) return null;
      entries.push({ isOob: false, configObus: new Uint8Array(0) });
    } else {
      const av2c = findChild(d, seStart + SAMPLE_ENTRY_HEADER_LEN, seEnd, 'av2C');
      if (!av2c) return null;
      const payload = d.subarray(av2c[0] + 8, av2c[1]);
      if (payload.length < AV2C_PREFIX_LEN) return null;
      const expectedCount = payload[1] + 1;
      let off = AV2C_PREFIX_LEN;
      for (let n = 0; n < expectedCount; n++) {
        let i = off;
        let obuSize = 0;
        let shift = 0;
        while (i < payload.length && i - off < 8) {
          const b = payload[i++];
          obuSize |= (b & 0x7f) << shift;
          if (!(b & 0x80)) break;
          shift += 7;
        }
        if (obuSize === 0 || i + obuSize > payload.length) return null;
        off = i + obuSize;
      }
      // Ignore any unrecognized trailing bytes after expectedCount config OBUs per ISOBMFF.
      entries.push({ isOob: true, configObus: payload.subarray(AV2C_PREFIX_LEN, off) });
    }
    seStart = seEnd;
  }
  return entries.length > 0 ? entries : null;
}

// Find the sample entry configs of the first AV2 ('av2E' or 'av2e') track in the file.
function extractConfigObus(d: Uint8Array): SampleEntryConfig[] | null {
  const moov = findChild(d, 0, d.length, 'moov');
  if (!moov) return null;
  let o = moov[0] + 8;
  while (o + 8 <= moov[1]) {
    const size = u32(d, o);
    const boxEnd = size === 0 ? moov[1] : o + size;
    if (size < 8 || boxEnd > moov[1]) break;
    if (fourCC(d, o + 4) === 'trak') {
      const cfg = configObusFromTrak(d, o + 8, boxEnd);
      if (cfg) return cfg;
    }
    o = boxEnd;
  }
  return null;
}

interface ParsedObuSlice {
  slice: Uint8Array;
  obuType: number;
  isPossiblyOob: boolean;
  isRapFrame: boolean;
}

function splitSampleObus(p: Uint8Array): ParsedObuSlice[] {
  const out: ParsedObuSlice[] = [];
  let off = 0;
  while (off + 2 <= p.length) {
    let i = off;
    let obuSize = 0;
    let shift = 0;
    while (i < p.length && i - off < 8) {
      const b = p[i++];
      obuSize |= (b & 0x7f) << shift;
      if (!(b & 0x80)) break;
      shift += 7;
    }
    if (obuSize === 0 || i + obuSize > p.length) break;
    const hdr0 = p[i];
    const extFlag = (hdr0 >> 7) & 0x01;
    const obuType = (hdr0 >> 2) & 0x1f;
    const hdrLen = extFlag ? 2 : 1;
    const xlayerId = extFlag && obuSize >= 2 ? p[i + 1] & 0x1f : 0;
    const isGlobalPrefixMeta =
      (obuType === 8 || obuType === 9) &&
      xlayerId === 31 &&
      obuSize > hdrLen &&
      ((p[i + hdrLen] >> 7) & 0x01) === 0;
    const isPossiblyOob =
      obuType === 1 ||
      obuType === 16 ||
      obuType === 17 ||
      obuType === 18 ||
      obuType === 20 ||
      isGlobalPrefixMeta;
    const isRapFrame = obuType === 4 || obuType === 5 || obuType === 21;
    out.push({
      slice: p.subarray(off, i + obuSize),
      obuType,
      isPossiblyOob,
      isRapFrame,
    });
    off = i + obuSize;
  }
  return out;
}

interface ExtractedSample {
  data: Uint8Array;
  descIdx: number;
  isSync: boolean;
}

// Reconstruct the elementary stream per av2-isobmff #av2-sample-reconstruction & #av2-bitstream-reconstruction.
// Mirrors the sample loop in av2_demuxer.cpp lines 323-389 (note that mp4box.js
// s.description_index is 0-based whereas libisomediafile desc_idx in
// av2_demuxer.cpp lines 247 and 327-343 is 1-based).
function reconstruct(samples: ExtractedSample[], configs: SampleEntryConfig[]): Uint8Array {
  const parts: Uint8Array[] = [];
  let prevDescIdx = -1;
  samples.forEach((sample, i) => {
    const cfg = configs[Math.max(0, Math.min(configs.length - 1, sample.descIdx))];
    const obus = splitSampleObus(sample.data);
    let startIdx = 0;
    // Step 1: Write Temporal Delimiter OBU.
    if (obus.length > 0 && obus[0].obuType === 2) {
      parts.push(obus[0].slice);
      startIdx = 1;
    } else {
      parts.push(TD_BYTES);
    }
    // Step 2: Prepend OutOfBandAV2CodecConfigurationBox OBUs at sample 0, sample entry changes, and RAPs.
    const hasRap = sample.isSync || obus.some(o => o.isRapFrame);
    if (cfg.isOob && (i === 0 || sample.descIdx !== prevDescIdx || hasRap) && cfg.configObus.length > 0) {
      parts.push(cfg.configObus);
    }
    prevDescIdx = sample.descIdx;
    // Step 3 & Step 4: Discard Possibly out-of-band OBUs in 'av2E' samples, write remaining OBUs.
    for (let k = startIdx; k < obus.length; k++) {
      if (cfg.isOob && obus[k].isPossiblyOob) continue;
      parts.push(obus[k].slice);
    }
  });
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// True if the bytes look like an ISOBMFF file (`....ftyp`) or the name is .mp4.
export function looksLikeMp4(data: Uint8Array, filename: string): boolean {
  if (/\.(mp4|m4s|m4v|mov)$/i.test(filename)) return true;
  return (
    data.length >= 8 &&
    data[4] === 0x66 && // 'f'
    data[5] === 0x74 && // 't'
    data[6] === 0x79 && // 'y'
    data[7] === 0x70 //    'p'
  );
}

// Demux an AV2 .mp4 into an .obu elementary stream.
export function demuxMp4ToObu(data: Uint8Array): Promise<Uint8Array> {
  const configObus = extractConfigObus(data);
  if (!configObus) {
    return Promise.reject(
      new Error("no AV2 sample entry ('av2E' or 'av2e') found in this file"),
    );
  }

  const mp4: any = createFile();

  return new Promise<Uint8Array>((resolve, reject) => {
    const collected: ExtractedSample[] = [];
    let expected = -1;
    let settled = false;

    const fail = (msg: string) => {
      if (settled) return;
      settled = true;
      reject(new Error(msg));
    };

    mp4.onError = (e: string) => fail(`mp4box error: ${e}`);

    mp4.onReady = (info: any) => {
      const track = (info.tracks || []).find(
        (t: any) =>
          typeof t.codec === 'string' &&
          (t.codec.startsWith('av2E') || t.codec.startsWith('av2e')),
      );
      if (!track) {
        fail("no AV2 ('av2E' or 'av2e') video track found in this file");
        return;
      }
      expected = track.nb_samples;
      if (expected === 0) {
        if (!settled) {
          settled = true;
          resolve(reconstruct([], configObus));
        }
        return;
      }
      mp4.setExtractionOptions(track.id, null, { nbSamples: expected });
      mp4.start();
    };

    mp4.onSamples = (_id: number, _user: unknown, samples: any[]) => {
      for (const s of samples) {
        collected.push({
          data: new Uint8Array(s.data),
          descIdx: s.description_index ?? 0,
          isSync: !!s.is_sync,
        });
      }
      if (!settled && expected >= 0 && collected.length >= expected) {
        settled = true;
        resolve(reconstruct(collected, configObus));
      }
    };

    // mp4box wants an ArrayBuffer tagged with its offset in the file.
    const ab = data.slice().buffer as ArrayBuffer & { fileStart?: number };
    ab.fileStart = 0;
    mp4.appendBuffer(ab);
    mp4.flush();
  });
}
