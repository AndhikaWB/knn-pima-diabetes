from knn import KNN

# TODO: nullify sebelum minmax
# Hilangkan koma bila integer

if __name__ == "__main__":
    # ==============================
    # Langkah 1
    # ==============================
    # Impor dataset dan label dari file CSV.
    # Dataset diperoleh dari PIMA Indians Diabetes.
    # https://www.kaggle.com/uciml/pima-indians-diabetes-database

    dataset = KNN.import_csv("dataset/diabetes_original.csv", True)
    # Cukup ambil 1 baris pertama untuk menyimpan label
    label = KNN.import_csv("dataset/diabetes_original.csv", False)[0]
    # Tampilkan 10 baris pertama pada dataset
    KNN.print_dataset(dataset, label, 10)
    input("Tekan ENTER untuk lanjut dari langkah 1...\n")

    # ==============================
    # Langkah Sampingan
    # ==============================
    # Simpan informasi kolom untuk proses selanjutnya.
    # Kolom perlu dibedakan tergantung dari tipe datanya.

    # Kolom penting yang nilainya mustahil nol pada kasus nyata
    # Meliputi kadar glukosa, tekanan darah, tebal kulit, dst
    indeks_kolom_penting = [1, 2, 3, 4, 5, 7]
    # Indeks kolom biasa selain kolom hasil
    indeks_kolom_biasa = [0, 1, 2, 3, 4, 5, 6, 7]
    # Indeks terakhir sekaligus hasil klasifikasi diabetes/sehat
    indeks_kolom_hasil = 8

    # ==============================
    # Langkah 2
    # ==============================
    # Ganti nilai nol pada kolom-kolom penting dengan null.
    # Perlu dilakukan untuk proses "imputasi" nantinya.
    # Lakukan ini sebelum menerapkan normalisasi rentang nilai
    # ke skala 0-1 untuk menjaga karakter data dalam dataset

    dataset = KNN.nullify_zero(dataset, indeks_kolom_penting)
    # Tampilkan 10 baris pertama pada dataset
    KNN.print_dataset(dataset, label, 10)
    input("Tekan ENTER untuk lanjut dari langkah 2...\n")
    
    # ==============================
    # Langkah 3
    # ==============================
    # Normalisasi rentang nilai pada dataset ke dalam skala 0-1.
    # Normalisasi nilai diperlukan untuk mengurangi bias jarak.
    # Dengan adanya normalisasi, jarak antara baris yang datanya
    # lengkap dengan baris yang datanya tidak lengkap dapat
    # diminimalisir sehingga dapat meningkatkan akurasi.

    # Cukup normalisasi kolom biasa karena kolom hasil sudah 0/1
    dataset = KNN.minmax_scaler(dataset, indeks_kolom_biasa)
    # Tampilkan 10 baris pertama pada dataset
    KNN.print_dataset(dataset, label, 10)
    input("Tekan ENTER untuk lanjut dari langkah 3...\n")

    # ==============================
    # Langkah 4
    # ==============================
    # Lakukan proses imputasi pada kolom-kolom penting dataset.
    # Imputasi adalah proses pengisian data kosong pada baris
    # dengan data-data yang diperoleh dari tetangga terdekat.
    # Baris terdekat bukan berarti tetangga terdekat!

    # Gunakan mean untuk mengukur nilai pada tetangga
    # Jumlah tetangga yang diperhitungkan adalah 5 tetangga terdekat
    dataset = KNN.imputer(dataset, indeks_kolom_penting, 5, "mean")
    KNN.print_dataset(dataset, label, 10)
    input("Tekan ENTER untuk lanjut dari langkah 4...\n")

    # ==============================
    # Langkah 5
    # ==============================
    # Lihat sebaran data untuk melihat ketidakseimbangan kelas
    # pada dataset. Dalam hal ini, jumlah penderita diabetes tentunya
    # lebih sedikit dari jumlah orang sehat. Oleh karena itu, perlu
    # dilakukan oversampling agar jumlah keduanya setara.
    # Oversampling adalah proses "duplikasi" data minoritas agar
    # jumlahnya setara dengan data mayoritas. Hasil duplikasi tidak
    # identik dengan data asalnya (sudah mengalami perubahan)

    # Bagi kelas berdasarkan nilai 0 (sehat) atau 1 (diabetes)
    dataset_sehat, dataset_diabetes = KNN.divide_dataset_class(dataset, [0, 1])

    print("* Sebelum oversampling *")
    print(f"Jumlah diabetes: {len(dataset_diabetes)}")
    print(f"Jumlah sehat: {len(dataset_sehat)}")

    # Lakukan oversampling dataset diabetes agar setara jumlah baris dataset sehat
    # Jumlah tetangga sebagai pertimbangan oversampling adalah 5 buah tetangga
    dataset_diabetes = KNN.smote_oversampling(list(dataset_diabetes), len(dataset_sehat), 5)

    print("* Sesudah oversampling *")
    print(f"Jumlah diabetes: {len(dataset_diabetes)}")
    print(f"Jumlah sehat: {len(dataset_sehat)}\n")

    input("Tekan ENTER untuk lanjut dari langkah 5...\n")

    # ==============================
    # Langkah 6
    # ==============================
    # Gabung kembali dataset sehat dan diabetes yang telah
    # dilakukan oversampling. Lalu, lakukan shuffle terhadap
    # dataset gabungan tersebut dengan harapan sebaran data
    # (kelas) pada dataset menjadi lebih natural. Natural
    # dalam artian baris diabetes/sehat tidak menumpuk
    # pada bagian tertentu dalam dataset

    # Gabung lalu shuffle dataset saat ini
    dataset = KNN.shuffle_dataset(dataset_diabetes + dataset_sehat)
    # Ekspor dataset yang telah di-shuffle untuk melihat sebaran data
    KNN.export_csv(dataset, "dataset/diabetes_result.csv", label, True)
    # Tampilkan 10 baris pertama pada dataset
    KNN.print_dataset(dataset, label, 10)
    input("Tekan ENTER untuk lanjut dari langkah 6...\n")

    # ==============================
    # Langkah 7
    # ==============================
    # Bagi dataset menjadi dataset training dan testing dan
    # lakukan uji akurasi. Bila dataset dibagi 5, maka 1/5
    # data akan dipakai sebagai data training, dan 4/5 
    # sisanya dipakai untuk testing. Proses ini dilakukan
    # 5 kali dengan posisi data testing yang berubah-ubah


    # Bagi dataset menjadi 5 bagian dan uji akurasi
    KNN.k_fold_crossval(dataset, 5)

    input("Tekan ENTER untuk melihat sampel dataset hasil...\n")
    KNN.print_dataset(dataset, label, 100)