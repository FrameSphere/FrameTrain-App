// Wahrnehmungs-Hash (dHash) fuer Beinahe-Dubletten bei Bildern.
//
// Das Bild wird auf 9x8 Graustufen verkleinert; jedes Bit sagt, ob ein Pixel
// heller ist als sein rechter Nachbar. Groesse, Kompression und Helligkeit
// aendern daran fast nichts — zwei Kopien desselben Fotos unterscheiden sich
// in wenigen Bits, zwei verschiedene Fotos in etwa der Haelfte.
//
// Gerechnet wird hier, weil der Webview jedes Bildformat dekodiert; das
// Backend muesste dafuer eine Bildbibliothek mitbringen. Es bekommt nur die
// fertigen Hashes (studio_save_hashes) und vergleicht sie.

/** 72 Graustufen (9 breit, 8 hoch, zeilenweise) -> 16 Hex-Zeichen. */
export function dhashFromGray(gray: ArrayLike<number>): string {
  if (gray.length !== 72) throw new Error(`dHash braucht 72 Werte, bekam ${gray.length}`);
  let hex = '';
  for (let row = 0; row < 8; row++) {
    let byte = 0;
    for (let col = 0; col < 8; col++) {
      const links = gray[row * 9 + col];
      const rechts = gray[row * 9 + col + 1];
      byte = (byte << 1) | (links > rechts ? 1 : 0);
    }
    hex += byte.toString(16).padStart(2, '0');
  }
  return hex;
}

export function hamming(a: string, b: string): number {
  let n = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i += 2) {
    let x = parseInt(a.slice(i, i + 2), 16) ^ parseInt(b.slice(i, i + 2), 16);
    while (x) { n += x & 1; x >>= 1; }
  }
  return n;
}

/** dHash eines Bildes unter einer URL (convertFileSrc).
 *
 *  Ueber fetch und createImageBitmap statt ueber ein <img>: ein Bild von der
 *  asset:-Adresse gilt dem Webview als fremde Herkunft, und das Canvas darf
 *  seine Pixel dann nicht herausgeben (getImageData wirft). In 1.3.3 scheiterte
 *  deshalb jeder Hash still, und die Pruefung meldete "keine Dubletten unter 0". */
export async function computeDHash(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Bild nicht lesbar (${res.status})`);
  const bitmap = await createImageBitmap(await res.blob());
  const canvas = document.createElement('canvas');
  canvas.width = 9;
  canvas.height = 8;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Kein Canvas-Kontext');
  ctx.drawImage(bitmap, 0, 0, 9, 8);
  bitmap.close?.();
  const rgba = ctx.getImageData(0, 0, 9, 8).data;
  const gray: number[] = [];
  for (let i = 0; i < rgba.length; i += 4) gray.push(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
  return dhashFromGray(gray);
}
