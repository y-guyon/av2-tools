// Demux an AV2-in-ISOBMFF (.mp4) file into an AV2 elementary bitstream (.obu)
// in the browser, so the existing OBU parser can process it unchanged.
//
// Mirrors the C++ av2_demux (av2-tools/apps/av2_demux/av2_demuxer.cpp):
//   - read the av2C configOBUs from every stsd sample entry (lines 157-187),
//   - for every sample emit a Temporal Delimiter, splice that sample's desc_idx
//     configOBUs after each sync sample's TD, then the sample's frame OBUs
//     (lines 204-269).
//
// mp4box.js handles ISOBMFF parsing + sample extraction. It does not know AV2,
// and mp4box 2.4.1 dropped the classic createBoxCtor registration API, so the
// one box it can't surface - av2C - is read here with a small manual box walk.

import { createFile } from 'mp4box';

// av2C payload layout (per av2-isobmff): a 9-byte prefix followed by configOBUs.
const AV2C_PREFIX_LEN = 9;
// VisualSampleEntry fixed fields preceding the child boxes (matches the C++
// kVisualSampleEntryFixedSize in av2_demuxer.cpp line 72): 8-byte box header + 78 bytes.
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

// Walk one trak's mdia > minf > stbl > stsd > av02 > av2C; return the per-entry
// configOBUs (av2C payload minus the 9-byte prefix), or null if not an AV2 track.
// Mirrors the stsd sample-entry loop in av2_demuxer.cpp lines 157-187.
function configObusFromTrak(d: Uint8Array, start: number, end: number): Uint8Array[] | null {
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
  const entries: Uint8Array[] = [];
  for (let e = 0; e < entryCount && seStart + 8 <= stsd[1]; e++) {
    const seSize = u32(d, seStart);
    const seEnd = seSize === 0 ? stsd[1] : Math.min(seStart + seSize, stsd[1]);
    if (seSize !== 0 && seSize < SAMPLE_ENTRY_HEADER_LEN) return null;
    if (fourCC(d, seStart + 4) !== 'av02') return null;
    const av2c = findChild(d, seStart + SAMPLE_ENTRY_HEADER_LEN, seEnd, 'av2C');
    if (!av2c) return null;
    const payload = d.subarray(av2c[0] + 8, av2c[1]);
    if (payload.length < AV2C_PREFIX_LEN) return null;
    entries.push(payload.subarray(AV2C_PREFIX_LEN));
    seStart = seEnd;
  }
  return entries.length > 0 ? entries : null;
}

// Find the per-sample-entry av2C configOBUs of the first AV2 (av02) track in the file.
function extractConfigObus(d: Uint8Array): Uint8Array[] | null {
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

// If `p` starts with an Annex-B framed Temporal Delimiter OBU, return its total
// length in bytes; otherwise 0. Mirrors leading_td_length() in the C++ demuxer
// (av2_demuxer.cpp lines 38-55).
function leadingTdLength(p: Uint8Array): number {
  if (p.length < 2) return 0;
  let i = 0;
  let obuSize = 0;
  let shift = 0;
  while (i < p.length && i < 8) {
    const b = p[i++];
    obuSize |= (b & 0x7f) << shift;
    if (!(b & 0x80)) break;
    shift += 7;
  }
  if (obuSize === 0 || i >= p.length || i + obuSize > p.length) return 0;
  const obuType = (p[i] >> 2) & 0x1f; // header byte: ext(1) type(5) tlayer(2)
  if (obuType !== 2) return 0; // not a TEMPORAL_DELIMITER
  return i + obuSize;
}

interface ExtractedSample {
  data: Uint8Array;
  descIdx: number;
  isSync: boolean;
}

// Reconstruct the elementary stream from per-sample-entry configOBUs + per-sample
// frame data. Mirrors the sample loop in av2_demuxer.cpp lines 204-269 (note that
// mp4box.js s.description_index is 0-based whereas libisomediafile desc_idx in
// av2_demuxer.cpp lines 160 and 222-244 is 1-based).
function reconstruct(samples: ExtractedSample[], configObusList: Uint8Array[]): Uint8Array {
  const parts: Uint8Array[] = [];
  samples.forEach((sample, i) => {
    const configObus =
      configObusList[Math.max(0, Math.min(configObusList.length - 1, sample.descIdx))];
    const isSync = sample.isSync || i === 0;
    const tdLen = leadingTdLength(sample.data);
    if (tdLen > 0) {
      parts.push(sample.data.subarray(0, tdLen)); // TD already in the sample
      if (isSync) parts.push(configObus);
      parts.push(sample.data.subarray(tdLen));
    } else {
      parts.push(TD_BYTES); // synthesize a TD at the sample boundary
      if (isSync) parts.push(configObus);
      parts.push(sample.data);
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
      new Error("no AV2 sample entry ('av02' with 'av2C') found in this file"),
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
        (t: any) => typeof t.codec === 'string' && t.codec.startsWith('av02'),
      );
      if (!track) {
        fail("no 'av02' (AV2) video track found in this file");
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
          // mp4box.js sample.description_index is 0-based (unlike 1-based desc_idx in
          // av2_demuxer.cpp lines 160 and 222-244).
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
