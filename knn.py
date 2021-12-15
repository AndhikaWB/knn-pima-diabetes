import csv
import math

class NoIndexFoundFromLabel(Exception):
    pass

class KNN:
    def __init__(self, train_set, label):
        # Dataset untuk dijadikan acuan bagi data baru
        self.train_set = self.impor_csv(train_set)
        # Nama-nama label kolom (header) untuk dataset training
        self.label = self.impor_csv(label)[0]

    # Ganti nilai nol ke "None" pada label kolom tertentu
    def _nol_ke_null(self, label, ganti = 0, dataset = None):
        # Gunakan dataset training bila parameter dataset kosong
        if dataset == None: dataset = self.train_set
        # Konversi label ke list jika hanya ada satu label
        if not isinstance(label, list): label = [label]

        indeks_label = []
        # Temukan indeks kolom dari tiap-tiap nama label
        for nama_label in label:
            for kol in range(len(self.label)):
                if self.label[kol] == nama_label:
                    indeks_label.append(kol)
        # Jangan lanjutkan bila indeks label kosong
        if indeks_label == []:
            raise NoIndexFoundFromLabel("tidak dapat menemukan indeks dari label")

        # Ganti nilai "ganti" pada kolom tertentu dengan "None"
        for bar in range(len(dataset)):
            for kol in range(len(dataset[bar])):
                if kol in indeks_label:
                    if dataset[bar][kol] == ganti:
                        dataset[bar][kol] = None
        # Kembalikan dataset yang sudah di-null-kan
        return dataset

    # Isi nilai "None" pada label (kolom) tertentu berdasarkan rata-rata K tetangga terdekat
    def _imputer(self, label, k_tetangga = 5, dataset = None):
        # Gunakan dataset training bila parameter dataset kosong
        if dataset == None: dataset = self.train_set

        # Konversi label ke list jika hanya ada satu label
        if not isinstance(label, list): label = [label]

        indeks_label = []
        # Temukan indeks kolom dari tiap-tiap label
        for nama_label in label:
            for kol in range(len(self.label)):
                if self.label[kol] == nama_label:
                    indeks_label.append([kol, nama_label])
        # Jangan lanjutkan bila indeks label kosong
        if indeks_label == []:
            raise NoIndexFoundFromLabel("tidak dapat menemukan indeks dari label")

        for bar in range(len(dataset)):
            for kol in range(len(dataset[bar])):
                if kol in [_[0] for _ in indeks_label]:
                    if dataset[bar][kol] == None:
                        for temp_bar in range(len(indeks_label)):
                            if kol == indeks_label[temp_bar][0]:
                                nama_kol = indeks_label[temp_bar][1]
                        tetangga = self.dapatkan_tetangga(dataset[bar], k_tetangga, dataset, nama_kol)
                        for data_kol in tetangga:
                            pass


    def impor_csv(self, in_file):
        # Baris dataset mula-mula
        dataset = []
        # Buka file CSV untuk dikonversi ke list dataset
        with open(in_file, "r") as file:
            for bar in csv.reader(file):
                for kol in bar:
                    try:
                        # Konversi string ke int atau float
                        kol = float(kol)
                        if kol.is_integer(): kol = int(kol)
                    # Biarkan string bila tidak bisa dikonversi
                    except ValueError: pass
                # Tambah baris saat ini ke dataset
                dataset.append(bar)
        # Kembalikan dataset yang bersih dari baris duplikat
        return list(dict.fromkeys(dataset))

    def ekspor_csv(self, out_file = "dataset.csv", dataset = None):
        # Gunakan dataset training bila parameter dataset kosong
        if dataset == None: dataset = self.train_set
        # Tulis baris-baris dataset ke dalam file
        with open(out_file, "w") as file:
            csv.writer(file).writerows(dataset)
    
    def print_dataset(self, jum_baris = 10, dataset = None):
        # Gunakan dataset training bila parameter dataset kosong
        if dataset == None: dataset = self.train_set
        # Panjang teks tiap-tiap nama label kolom
        pjg_label = []
        # Tampilkan label
        for kol in self.label:
            # Simpan panjang label untuk dipakai nanti
            pjg_label.append(len(kol) + 4)
            # Tetapkan lebar kolom sesuai panjang label + 3
            print(f"{kol:<{len(kol) + 4}}", end = " ")
        print()
        # Tampilkan dataset
        for i, bar in enumerate(dataset):
            if i == jum_baris - 1: break
            for j, kol in enumerate(bar):
                # Tetapkan lebar kolom sesuai panjang label + 3
                print(f"{kol:<{pjg_label[j]}}", end = " ")
            print()

    def ukur_jarak(self, baris1, baris2):
        # Jarak mula-mula antar baris data
        jarak = 0.0
        # Iterasi kolom-kolom pada baris data
        for i in range(len(baris1) - 1):
            # Ganti nilai "None" dengan nol bila ada
            if baris1[i] == None: bar1 = 0
            else: bar1 = baris1[i]
            if baris2[i] == None: bar2 = 0
            else: bar2 = baris2[i]
            # Rumus jarak euklides = akar dari [ (x1-x2)^2 + (y1-y2)^2 + ... ]
            jarak += (bar1 - bar2[i]) ** 2
        # Kembalikan akar dari jarak sebelumnya
        return math.sqrt(jarak)
    
    def dapatkan_tetangga(self, test_bar, k_tetangga = 5, dataset = None, filter_label = None, filter_nilai = None):
        # Gunakan dataset training bila parameter dataset kosong
        if dataset == None: dataset = self.train_set

        # Jarak antara 1 "test_bar" dengan semua data pada dataset
        # Variabel "test_bar" tidak boleh beranggota lebih dari 1 baris
        jarak = []
        for bar in dataset:
            # Jarak "baris" tes dengan suatu baris lainnya pada dataset
            temp_jarak = self.ukur_jarak(bar, test_bar)
            # Tambahkan data baris beserta jaraknya ke "jarak"
            jarak.append([bar, temp_jarak])
        # Sortir "jarak" berdasarkan kolom ke-1 (jarak terpendek)
        jarak.sort(key=lambda x: x[1])

        # Jika tidak diterapkan "filter_label" maka langsung return
        if filter_label == None:
            # Kembalikan K tetangga pertama (tanpa informasi jarak)
            return jarak[0][:k_tetangga]
        else:
            # Konversi "filter_label" ke list jika hanya ada satu label
            if not isinstance(filter_label, list): filter_label = [filter_label]

            indeks_label = []
            # Temukan indeks kolom dari tiap-tiap "filter_label"
            for nama_label in filter_label:
                for kol in range(len(self.label)):
                    if self.label[kol] == nama_label:
                        indeks_label.append(kol)
            # Jangan lanjutkan bila indeks label kosong
            if indeks_label == []:
                raise NoIndexFoundFromLabel("tidak dapat menemukan indeks dari label")

            # Tetangga terdekat mula-mula
            tetangga = []
            for bar in jarak:
                ketemu_filter = False
                for kol in range(len(bar[0])):
                    # Cek apakah nilai pada "indeks_label" ada yang bernilai "filter_nilai"
                    if kol in indeks_label:
                        if bar[kol] == filter_nilai:
                            ketemu_filter = True
                            break
                # Jika "filter_nilai" tidak ditemukan, tambahkan baris ke list tetangga
                if not ketemu_filter:
                    tetangga.append(bar[0])
                    if len(tetangga) >= k_tetangga: break
            # Kembalikan tetangga terdekat yang telah di-filter
            return tetangga

    def prediksi_kelas(self, test_set, k_tetangga = 5):
        # Jarak mula-mula antara "baris" dengan semua data "train_set"
        # "baris" hanya berupa 1 baris data, bukan 1 dataset
        jarak = []
        for bar in range(len(self.train_set)):
            # Jarak "baris" dengan salah satu baris pada "train_set"
            temp_jarak = self.ukur_jarak(self.train_set)

if __name__ == "__main__":
    diabetes = KNN("train/diabetes.csv", "train/diabetes.txt")
    diabetes.print_dataset()