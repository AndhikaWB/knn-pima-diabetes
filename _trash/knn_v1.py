# Referensi:
# https://towardsdatascience.com/lets-make-a-knn-classifier-from-scratch-e73c43da346d
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import csv
import math

class KNN:
    def __init__(self, dataset, label):
        self.dataset = self.import_csv(dataset)
        self.label = self.import_csv(label)[0]

    # Impor CSV menjadi array multi dimensi
    def import_csv(self, file):
        # Total baris mula-mula
        all_rows = []
        # Buka file CSV dalam mode baca
        with open(file, "r") as f:
            for row in csv.reader(f):
                # Baris saat ini
                curr_row = []
                for col in row:
                    # Konversi string kolom saat ini ke int atau float
                    col = self.__string_to_float(col)
                    # Tambahkan kolom saat ini ke baris saat ini
                    curr_row.append(col)
                # Tambahkan baris saat ini ke total baris
                all_rows.append(curr_row)
        # Kembalikan total baris
        return all_rows

    # Konversi string ke int atau float (bila memungkinkan)
    def __string_to_float(self, value):
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except: pass
        return value
    
    def __normalize_data(self):
        # Hapus duplikat
        # KNN Imputer
        # Min max scale?
        pass
    
    # Tampilkan dataset training
    def print_dataset(self, num_rows = 10):
        label_lengths = []
        # Tampilkan label
        for col in self.label:
            label_lengths.append(len(col)+3)
            print(f"{col:>{len(col)+3}}", end = " ")
        print()
        # Tampilkan data
        for i, row in enumerate(self.dataset):
            if i == num_rows - 1: break
            for j, col in enumerate(row):
                # Sesuaikan lebar kolom dengan panjang teks label
                print(f"{col:>{label_lengths[j]}}", end = " ")
            print()

    # Dapatkan jarak euklides antara 2 baris data
    def get_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)
    
    # Prediksi golongan klasifikasi untuk data baru
    def get_prediction(self, test_dataset, k_neighbors = 5):
        distances = []
        for i in range(len(self.dataset)):
            curr_dist = self.get_distance(self.dataset[i][:-1], test_dataset)
            distances.append([self.dataset[i], curr_dist])
        # Sortir list berdasarkan jarak (kolom ke-1)
        distances.sort(key=lambda x: x[1])

        neighbors = []
        for i in range(k_neighbors):
            neighbors.append(distances[i][0])
        
        classes = [results[-1] for results in neighbors]
        # Kembalikan prediksi berupa modus (kelas terbanyak tetangga)
        return max(classes, key = classes.count)

    def get_accuracy(actual, predicted):
        corrects = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                corrects += 1
        return corrects / len(actual)

if __name__ == "__main__":
    knn = KNN("train/diabetes.csv", "train/diabetes.txt")
    knn.print_dataset()