## 7. Pembahasan hasil eksperimen

Eksperimen dilaksanakan pada 6 November 2026 menggunakan TensorFlow 2.21.0 dengan
GPU yang terdeteksi pada runtime Colab. CNN dilatih dari awal pada citra RGB 32×32,
menggunakan Adam dengan learning rate 0,001, batch size 128, dan seed 42. FAKE
ditetapkan sebagai kelas positif dan threshold klasifikasi sebesar 0,5.

### Pembagian dan audit data

| Split | REAL | FAKE | Total |
|---|---:|---:|---:|
| Train | 40.000 | 38.961 | 78.961 |
| Validation | 10.000 | 9.741 | 19.741 |
| Test | 10.000 | 10.000 | 20.000 |

Dari 100.000 gambar train resmi, 378 gambar disisihkan karena pikselnya identik
dengan gambar test. Selanjutnya, 920 pengulangan piksel identik dalam train
disisihkan. Sisa 98.702 gambar dibagi secara terstratifikasi menjadi train dan
validation. Bagian test resmi dipertahankan seluruhnya; audit mencatat 38
pengulangan piksel di dalam test. Pengujian ini masih memakai pembobotan per file
pada test resmi, sehingga pengulangan tersebut ikut dihitung. Audit ini memeriksa
kesamaan piksel RGB secara tepat, belum kemiripan visual atau citra turunan.

### Pelatihan dan pemilihan checkpoint

Pelatihan berjalan selama 14 dari maksimum 20 epoch. Validation loss terendah
tercatat pada **epoch 10**, yaitu **0,1389**, dengan validation accuracy **94,78%**.
Checkpoint inilah yang dimuat untuk evaluasi test. Validation accuracy tertinggi
sebesar 95,16% muncul pada epoch 12, tetapi kriteria pemilihan model adalah
validation loss, sehingga checkpoint yang digunakan tetap epoch 10.

Setelah epoch 10, training loss menurun dari 0,0924 menjadi 0,0602 pada epoch 14,
sedangkan validation loss meningkat menjadi 0,1714. Pola ini mengindikasikan
overfitting pada tahap akhir. Early stopping menghentikan pelatihan setelah
empat epoch tanpa perbaikan validation loss. Nilai per epoch di sini mengikuti
log notebook yang dibulatkan sampai empat desimal.
Pada grafik yang sumbu epochnya dimulai dari 0, epoch 10 pada log berada di posisi 9.

### Evaluasi pada test

| Metrik | Hasil |
|---|---:|
| Accuracy | **94,805%** |
| Precision FAKE | **95,86%** |
| Recall FAKE | **93,66%** |
| F1-score FAKE | **94,74%** |
| ROC-AUC | **0,9886** |
| Accuracy baseline mayoritas | **50,00%** |

CNN mengklasifikasikan 18.961 dari 20.000 gambar test dengan benar. Accuracy lebih
tinggi 44,805 poin persentase dibandingkan baseline mayoritas yang dipilih dari
train. Perbandingan ini mendukung bahwa model mempelajari pola yang berguna pada
CIFAKE; hasilnya belum menunjukkan generalisasi ke semua sumber gambar AI.

Confusion matrix, dengan baris label sebenarnya dan kolom prediksi:

| Label sebenarnya | Prediksi REAL | Prediksi FAKE |
|---|---:|---:|
| REAL | 9.595 | 405 |
| FAKE | 634 | 9.366 |

Sebanyak 405 gambar REAL salah ditandai sebagai FAKE, yaitu 4,05% dari seluruh
gambar REAL. Sebanyak 634 gambar FAKE lolos sebagai REAL, yaitu 6,34% dari seluruh
gambar FAKE. Dengan threshold 0,5, model lebih sering melewatkan gambar FAKE
daripada salah menandai gambar REAL. Precision FAKE sebesar 95,86% berarti
sebagian besar gambar yang ditandai FAKE memang berlabel FAKE pada test ini.

### Kesalahan dan keterbatasan

Galeri tersimpan menampilkan delapan gambar REAL yang salah ditandai sebagai
FAKE, termasuk satu gambar dengan skor FAKE 0,981. Ini menunjukkan bahwa skor
yang tinggi tetap dapat menghasilkan prediksi keliru. Contoh tersebut merupakan
delapan kesalahan pertama menurut urutan manifest, bukan sampel acak seluruh
kesalahan. Karena manifest test disusun dari REAL lalu FAKE, galeri didominasi
kesalahan kelas REAL. Galeri ini tidak cukup untuk menyimpulkan pola keseluruhan
kelas yang sulit.

Resolusi 32×32 membatasi detail visual yang tersedia. Kemiripan tekstur atau pola
lokal antarkelas merupakan kemungkinan penyebab kesalahan, tetapi belum dibuktikan
dengan analisis fitur. Model juga dapat mempelajari karakteristik sumber data atau
kompresi. Penelitian ini belum melakukan Grad-CAM, pengujian generator lain,
analisis citra yang diedit, maupun pengujian citra media sosial.

Satu usulan perbaikan adalah menguji checkpoint yang sudah dibekukan pada dataset
terpisah dari generator lain, dengan asal gambar serta label yang terdokumentasi.
Hasil uji tersebut perlu dilaporkan terpisah dari test CIFAKE. Perubahan arsitektur
atau regularisasi selanjutnya harus dipilih melalui validation, tanpa menyesuaikan
model berulang kali berdasarkan test.

### Kesimpulan

CNN sederhana dengan 355.649 parameter dapat membedakan gambar asli dan sintetis
AI dalam distribusi CIFAKE, dengan accuracy test 94,805% dan ROC-AUC 0,9886. Model
terbaik dipilih pada epoch 10 berdasarkan validation loss, sebelum muncul pola
overfitting pada epoch selanjutnya. Kesimpulan dibatasi pada dataset dan pengaturan
eksperimen ini; skor model bukan bukti pasti asal-usul suatu gambar.
