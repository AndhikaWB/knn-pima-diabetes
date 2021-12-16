from knn import KNN

if __name__ == "__main__":
    # Impor label dan dataset training
    diabetes_label = KNN.import_csv("train/label.txt", False)[0]
    diabetes_training = KNN.import_csv("train/diabetes.csv", False)

    # Import dataset testing (belum diklasifikasi) beserta jawabannya
    diabetes_testing = KNN.import_csv("test/diabetes.csv", False)
    diabetes_answer = KNN.import_csv("test/answer.txt", False)

    # Tampilkan 10 baris pertama pada tiap-tiap dataset
    print("* Dataset training mula-mula *")
    KNN.print_dataset(diabetes_training, diabetes_label, 10)
    print("* Dataset testing mula-mula *")
    KNN.print_dataset(diabetes_testing, diabetes_label, 10)

    # Gabung dataset training dengan dataset testing
    diabetes_gabungan = diabetes_training + diabetes_testing
    # Kolom penting yang nilainya tidak boleh nol
    indeks_kolom_penting = [1,2,3,4,5,7]
    # Indeks kolom biasa selain kolom klasifikasi
    indeks_kolom_biasa = [0,1,2,3,4,5,6,7]
    # Kolom klasifikasi penyakit (diabetes atau sehat)
    indeks_kolom_hasil = 8

    # Normalisasi rentang data menjadi 0-1 untuk mengurangi bias jarak
    diabetes_gabungan = KNN.minmax_scaler(diabetes_gabungan, indeks_kolom_biasa, 4)
    print("* Dataset gabungan setelah nilai diskalakan ke 0-1 *")
    KNN.print_dataset(diabetes_gabungan, diabetes_label, 10)

    # Ganti nilai nol pada kolom-kolom penting dengan null
    diabetes_gabungan = KNN.nullify_zero(diabetes_gabungan, indeks_kolom_penting)
    print("* Dataset gabungan setelah nilai nol diganti null *")
    KNN.print_dataset(diabetes_gabungan, diabetes_label, 10)

    # Isi nilai null dengan rata-rata nilai dari 5 tetangga terdekat
    diabetes_gabungan = KNN.imputer(diabetes_gabungan, indeks_kolom_penting, 5, "mean", 4)
    print("* Dataset gabungan setelah nilai null diimputasi *")
    KNN.print_dataset(diabetes_gabungan, diabetes_label, 10)

    # Tes akurasi pengklasifikasian ke sesama data dalam dataset training
    # Bagi dataset training menjadi 5 bagian, lalu tes 5 kali dengan tetangga terdekat 5 buah
    #KNN.k_fold_crossval(diabetes_training + diabetes_answer, 5, 5)

    # Pisah kembali dataset gabungan menjadi dataset training dan dataset testing
    diabetes_training = diabetes_gabungan[:len(diabetes_training)] # Sisakan dataset training
    diabetes_testing = diabetes_gabungan[-len(diabetes_testing):] # Sisakan dataset testing
    
    # Tes regresi untuk menentukan nilai tetangga yang paling optimal
    #KNN.regression_test(diabetes_testing, diabetes_training, diabetes_answer)

    # Prediksi akurasi klasifikasi berdasarkan nilai modus dari 27 tetangga terdekat (optimal berdasarkan regresi)
    print("* Dataset testing setelah diklasifikasi *")
    diabetes_testing = KNN.predict_class(diabetes_testing, diabetes_training, diabetes_label, 27, 10, diabetes_answer)

    # Ekspor hasil akhir testing ke dalam file CSV kembali
    KNN.export_csv(diabetes_testing, "result/diabetes.csv", None, True)