# Implementasi Algoritma A* dalam Segmentasi Citra
## Identitas Pengembang

Derwin Rustanly / 13522115 

## Deskripsi

Segmentasi citra adalah langkah penting dalam pengolahan citra yang mempengaruhi berbagai aplikasi, mulai dari pengenalan objek hingga analisis medis. Algoritma A* berbasis heuristik digunakan untuk memperhalus batas-batas klaster yang dihasilkan oleh K-means clustering, menghasilkan segmentasi yang lebih akurat dan konsisten.

Repository ini berisi implementasi algoritma A* untuk memperbaiki hasil segmentasi citra yang diperoleh melalui algoritma K-means. Algoritma ini diterapkan pada citra grayscale dan berwarna untuk meningkatkan akurasi dan kejelasan batas-batas klaster.

## Fitur

- Implementasi algoritma K-means untuk segmentasi awal citra.
- Penggunaan algoritma A* untuk memperbaiki batas-batas klaster hasil segmentasi K-means.
- Evaluasi hasil segmentasi menggunakan metrik Dice Coefficient dan Jaccard Index.
- Visualisasi hasil segmentasi sebelum dan sesudah perbaikan.

## Struktur Direktori
```
ImplementasiAStardalamSegmentasiCitra/
│
├── results/ 
├── src/ 
│ ├── Algo.py
│ └── SegmentationImage.py
├── test/ 
├── README.md 
```

## Dependensi
1. Python 3.x
2. OpenCV
3. NumPy
4. Matplotlib

## Instalasi dan Tata Cara Penggunaan program

1. Clone repository ini:
   ```bash
   git clone https://github.com/DerwinRustanly/ImplementasiAStardalamSegmentasiCitra.git
   ```
2. Masuk ke direktori proyek:
    ```bash
    cd ImplementasiAStardalamSegmentasiCitra
    ```
3. Lakukan instalasi dependesi yang dibutuhkan, dianatara MatPlotLib dan OpenCV
4. Jalankan perintah berikut:
    ```bash
    python src/SegmentationImage.py
    ```