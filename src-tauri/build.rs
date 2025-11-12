use std::{env, fs, io::Write, path::PathBuf};

fn b64_to_bytes(s: &str) -> Result<Vec<u8>, String> {
  fn val(c: u8) -> Option<u8> {
    match c {
      b'A'..=b'Z' => Some(c - b'A'),
      b'a'..=b'z' => Some(c - b'a' + 26),
      b'0'..=b'9' => Some(c - b'0' + 52),
      b'+' => Some(62),
      b'/' => Some(63),
      _ => None,
    }
  }
  let bytes = s.as_bytes();
  let mut out = Vec::with_capacity(s.len() * 3 / 4);
  let mut i = 0;
  while i < bytes.len() {
    // skip whitespace
    while i < bytes.len() && (bytes[i] == b'\r' || bytes[i] == b'\n' || bytes[i] == b' ' || bytes[i] == b'\t') { i += 1; }
    if i >= bytes.len() { break; }
    let a = bytes[i]; i += 1;
    let b = bytes.get(i).copied().unwrap_or(b'='); i += 1;
    let c = bytes.get(i).copied().unwrap_or(b'='); i += 1;
    let d = bytes.get(i).copied().unwrap_or(b'='); i += 1;
    if a == b'=' || b == b'=' { break; }
    let v0 = val(a).ok_or("invalid base64")? as u32;
    let v1 = val(b).ok_or("invalid base64")? as u32;
    let v2 = if c == b'=' { None } else { Some(val(c).ok_or("invalid base64")? as u32) };
    let v3 = if d == b'=' { None } else { Some(val(d).ok_or("invalid base64")? as u32) };
    out.push(((v0 << 2) | (v1 >> 4)) as u8);
    if let Some(v2) = v2 {
      out.push((((v1 & 0x0F) << 4) | (v2 >> 2)) as u8);
      if let Some(v3) = v3 {
        out.push((((v2 & 0x03) << 6) | v3) as u8);
      }
    }
  }
  Ok(out)
}

fn ensure_icon() -> Result<(), String> {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?;
  let mut dir = PathBuf::from(manifest_dir);
  dir.push("icons");
  let mut icon_path = dir.clone();
  icon_path.push("icon.ico");
  if icon_path.exists() { return Ok(()); }
  fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

  // Generate a minimal 16x16 32-bit ARGB (all transparent) ICO with BMP header.
  // ICONDIR
  let mut data: Vec<u8> = Vec::new();
  data.extend_from_slice(&[0x00,0x00, 0x01,0x00, 0x01,0x00]); // header
  // ICONDIRENTRY
  let bytes_in_res: u32 = 40 + (16*16*4) + 64; // BITMAPINFOHEADER + pixel data + AND mask
  data.push(0x10); // width 16
  data.push(0x10); // height 16
  data.push(0x00); // colors
  data.push(0x00); // reserved
  data.extend_from_slice(&[0x01,0x00]); // planes
  data.extend_from_slice(&[0x20,0x00]); // bitcount 32
  data.extend_from_slice(&bytes_in_res.to_le_bytes());
  data.extend_from_slice(&22u32.to_le_bytes()); // image offset
  // BITMAPINFOHEADER
  data.extend_from_slice(&40u32.to_le_bytes()); // header size
  data.extend_from_slice(&16i32.to_le_bytes()); // width
  data.extend_from_slice(&(16i32*2).to_le_bytes()); // height *2 (including AND mask)
  data.extend_from_slice(&[0x01,0x00]); // planes
  data.extend_from_slice(&[0x20,0x00]); // bitcount
  data.extend_from_slice(&[0x00,0x00,0x00,0x00]); // compression BI_RGB
  let image_bytes: u32 = 16*16*4;
  data.extend_from_slice(&image_bytes.to_le_bytes()); // size image
  data.extend_from_slice(&[0x00,0x00,0x00,0x00]); // x ppm
  data.extend_from_slice(&[0x00,0x00,0x00,0x00]); // y ppm
  data.extend_from_slice(&[0x00,0x00,0x00,0x00]); // clr used
  data.extend_from_slice(&[0x00,0x00,0x00,0x00]); // clr important
  // Pixel data (all zeros = transparent)
  data.extend(std::iter::repeat(0u8).take(image_bytes as usize));
  // AND mask (16 rows * padded 4 bytes per row = 64 bytes)
  data.extend(std::iter::repeat(0u8).take(64));

  let mut f = fs::File::create(&icon_path).map_err(|e| e.to_string())?;
  f.write_all(&data).map_err(|e| e.to_string())?;

  // Also create a small PNG for bundler icons
  let mut png_path = dir.clone();
  png_path.push("app.png");
  if !png_path.exists() {
    let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9yQAvGQAAAAASUVORK5CYII=";
    let png = b64_to_bytes(png_b64)?;
    let mut pf = fs::File::create(&png_path).map_err(|e| e.to_string())?;
    pf.write_all(&png).map_err(|e| e.to_string())?;
  }
  Ok(())
}

fn main() {
  let _ = ensure_icon();
  // Warn if CLI executable missing; MSI/resources will omit it otherwise.
  if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
    let mut cli = PathBuf::from(&manifest_dir);
    cli.push("..");
    cli.push("file_renamer.exe");
    if !cli.exists() {
      println!("cargo:warning=CLI executable not found at {}. The MSI/portable may miss file_renamer.exe. Build it or place it at repository root.", cli.display());
    }
  }
  tauri_build::build()
}
