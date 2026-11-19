import os
import io
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, ImageOps, ImageEnhance

# Muat GEMINI_API_KEY dari file .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY belum diisi di dalam file .env!")

genai.configure(api_key=api_key)

app = FastAPI(title="Bali Puspa Laundry - AI Service")


def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Ubah foto nota ke grayscale dan tingkatkan kontras agar tulisan lebih jelas terbaca."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.grayscale(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Server AI Bali Puspa Laundry aktif!"}


@app.post("/api/v1/scan-nota")
async def scan_nota(file: UploadFile = File(...)):
    # Validasi awal: pastikan file yang diunggah adalah gambar
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Format file tidak valid. Harap unggah foto nota (JPG/PNG)."
        )

    try:
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)

        model = genai.GenerativeModel("gemini-3.8-flash")
        prompt = """
        Kamu adalah asisten pencatat nota untuk usaha 'Bali Puspa Laundry'.
        Baca foto nota transaksi ini dan ekstrak informasinya ke dalam format JSON murni tanpa teks tambahan.
        Gunakan struktur JSON berikut:
        {
          "tanggal": "YYYY-MM-DD atau null jika tidak terbaca",
          "nama_pelanggan": "nama pelanggan/villa/hotel atau null",
          "items": [
            {
              "teks_asli": "nama layanan persis seperti tertulis di nota",
              "qty": 1,
              "satuan": "kg atau pcs"
            }
          ]
        }
        """

        response = model.generate_content([prompt, processed_img])
        raw_text = response.text.strip()

        # Bersihkan pembungkus markdown ```json jika muncul
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").replace("json", "", 1).strip()

        hasil_ekstraksi = json.loads(raw_text)

        return {
            "status": "success",
            "data": hasil_ekstraksi
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="AI gagal memformat hasil bacaan nota ke JSON. Silakan foto ulang atau input manual."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Gagal memproses gambar nota: {str(e)}"
        )