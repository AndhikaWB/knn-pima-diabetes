"""
* Author *
  Andhika Putra Pratama (119140224)
  Andhika Wibawa Bhagaskara (119140218)

* References *
  https://towardsdatascience.com/lets-make-a-knn-classifier-from-scratch-e73c43da346d
  https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch
  https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning
  https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79
  https://colab.research.google.com/drive/1ptnsiqq4aX-YNYfxdpBPgbQolQrrEjeJ

* GitHub *
  https://github.com/AndhikaWB/knn-pima-diabetes
"""

# For reading and writing CSV file
import csv
# For using mean, median, and mode
import statistics

class KNN:
    @staticmethod
    def import_csv(input_file, header = True):
        dataset = []
        # Open input file (read mode)
        with open(input_file, "r") as file:
            # Skip the first row in file if header exist
            if header: reader = list(csv.reader(file)[1:])
            else: reader = list(csv.reader(file))
            for row in range(len(reader)):
                # Read column for each row
                for col in range(len(reader[row])):
                    try:
                        if reader[row][col] in ("None", "NaN"):
                            # Convert "empty" string in column to null (None)
                            reader[row][col] = None
                        else:
                            # Convert other string in column to float or int
                            reader[row][col] = float(reader[row][col])
                            if reader[row][col].is_integer():
                                reader[row][col] = int(reader[row][col])
                    except ValueError:
                        # Do not convert string if not possible
                        pass
                # Append converted row to dataset
                dataset.append(reader[row])
        # Return dataset with duplicate rows removed
        unique_dataset = []
        for row in dataset:
            if row not in unique_dataset:
                unique_dataset.append(row)
        return unique_dataset

    @staticmethod
    def export_csv(dataset, output_file, header_list = None, on_linux = False):
        # Workaround for extra blank lines on Windows
        nl = "\n" if on_linux else ""
        # Open output file (write mode)
        with open(output_file, "w", newline = nl) as file:
            if header_list != None:
                # Write header too if header list is specified
                csv.writer(file).writerow(header_list)
            # Write rows from list in dataset
            csv.writer(file).writerows(dataset)

    @staticmethod
    # Header list can't be empty since column size is based on that
    # Extra column spaces are useful when column values are longer than header
    def print_dataset(dataset, header_list, max_rows = 10, extra_column_spaces = 4):
        # Cancel print (useful in some scenarios)
        if max_rows < 1: return
        header_lengths = []
        # Print dataset header
        for header in header_list:
            # Calculate header length plus extra column spaces
            header_lengths.append(len(header) + extra_column_spaces)
            print(f"{header:<{header_lengths[-1]}}", end = " ")
        print()
        # Print data in dataset
        for row in range(len(dataset)):
            # Stop if it reached maximum rows from parameter
            if row >= max_rows: break
            for col in range(len(dataset[row])):
                try:
                    # Print each column with same width as header
                    print(f"{dataset[row][col]:<{header_lengths[col]}}", end = " ")
                except TypeError:
                    # Workaround for null (None) data type
                    print(f'{"NaN":<{header_lengths[col]}}', end = " ")
            print()
        print()

    # ==============================
    # Single row operation
    # ==============================

    @staticmethod
    # Get euclidean distance between 2 data rows
    # data_row1 must not contain null (None) value
    # data_row2 can contain some null values
    # Null columns are ignored to estimate distance (may increase accuracy)
    # Return negative distance if data_row1 contain null value
    # Compatible rows should always return positive distance
    def get_distance(data_row1, data_row2):
        # Copy list as new object, not as reference
        ndata_row1 = data_row1[:]
        ndata_row2 = data_row2[:]
        # Initial distance
        distance = 0.0
        # Iterate columns to sum distance except last column (class/result column)
        for col in range(len(data_row1) - 1):
            # Immediately return if value from data_row1 is null
            if ndata_row1[col] == None: return -1
            # Skip column if value from data_row2 is null
            if ndata_row2[col] == None: continue
            # Calculate euclidean distance (without square root)
            distance += (ndata_row1[col] - ndata_row2[col]) ** 2
        # Return square root of distance
        return distance ** 0.5

    @staticmethod
    # Get closest neighbors of data row from dataset. Filter colums and value must be used together
    # For example, if filter_columns_index = [0, 1, 2] and filter value = None
    # Then the neighbors will not have index 0, 1, 2 with null (None) value
    def get_neighbors(data_row, dataset, k_neighbors = 5, filter_columns_index = None, filter_value = None):
        distances = []
        for row in dataset:
            # Get distance from a row (in dataset) to data row (from parameter)
            curr_distance = KNN.get_distance(row, data_row)
            # Save row data and the distance
            # Do not append if both rows incompatible
            if curr_distance >= 0:
                # Add row as copy, just in case passed as reference
                distances.append([row[:], curr_distance])
        # Sort neighbors by closest distance (column index 1)
        distances.sort(key = lambda x: x[1])
        # Begin returning or filtering data
        if filter_columns_index == None:
            # Return row data (column index 0) for the first K neighbors
            # No need to continue filtering data
            return [row[0] for row in distances[:k_neighbors]]
        else:
            # Convert filter_columns_index to list even if there's only one column
            if not isinstance(filter_columns_index, list):
                filter_columns_index = [filter_columns_index]
            neighbors = []
            # Begin filtering column(s) by filter value
            for neighbor in range(len(distances)):
                found_filter_value = False
                # Get row data only (column index 0), not including distance
                for col in range(len(distances[neighbor][0])):
                    if col in filter_columns_index:
                        # Check if column contain filtered value
                        # If true then don't add this neighbor
                        if distances[neighbor][0][col] == filter_value:
                            found_filter_value = True
                            break
                if not found_filter_value:
                    # Append row data only (column index 0), not including distance
                    neighbors.append(distances[neighbor][0])
                    if len(neighbors) >= k_neighbors: break
            # Return filtered neighbors
            return neighbors
    
    # ==============================
    # Multi rows operation
    # ==============================

    @staticmethod
    # Convert dataset value to range between 0 (min) - 1 (max)
    # Can reduce bias when calculating distance
    # May also increase performance slightly
    def minmax_scaler(dataset, filter_columns_index, precision = 4):
        # Copy list as new object, not as reference
        ndataset = dataset[:]
        # Save min and max value of each column
        minmax_values = []
        for col in range(len(ndataset[0])):
            if col in filter_columns_index:
                # Get column values for all rows
                col_values = [row[col] for row in ndataset]
                min_value = min(col_values)
                max_value = max(col_values)
                # Append min and max value to list
                minmax_values.append([min_value, max_value])
            else: minmax_values.append([None, None])
        # Begin scaling column value
        for row in range(len(ndataset)):
            for col in range(len(ndataset[row])):
                if col in filter_columns_index:
                    # Scale column value to range between 0-1
                    ndataset[row][col] = ndataset[row][col] - minmax_values[col][0]
                    ndataset[row][col] /= minmax_values[col][1] - minmax_values[col][0]
                    # Round column value if needed
                    if precision:
                        ndataset[row][col] = round(ndataset[row][col], precision)
        # Return scaled dataset
        return ndataset

    @staticmethod
    # Convert zero to null (None) on specific column(s)
    def nullify_zero(dataset, filter_columns_index, zero_value = 0):
        # Copy list as new object, not as reference
        ndataset = dataset[:]
        # Convert columns index to list even if there's only one column
        if not isinstance(filter_columns_index, list):
            filter_columns_index = [filter_columns_index]
        # Read dataset as rows and columns
        for row in range(len(ndataset)):
            for col in range(len(ndataset[row])):
                # Replace zero to null (None) if index matches
                if col in filter_columns_index:
                    if ndataset[row][col] == zero_value:
                        ndataset[row][col] = None
        # Return dataset with zero replaced with null
        return ndataset

    @staticmethod
    # Replace null (None) value on dataset with data from neighbors
    # Use nullify first to convert zero to null in specific columns
    def imputer(dataset, filter_columns_index, k_neighbors = 5, mode = "mean", mean_precision = None):
        # Copy list as new object, not as reference
        ndataset = dataset[:]
        # Convert columns index to list even if there's only one column
        if not isinstance(filter_columns_index, list):
            filter_columns_index = [filter_columns_index]
        # Iterate rows and columns in dataset
        for row in range(len(ndataset)):
            for col in range(len(ndataset[row])):
                if col in filter_columns_index:
                    if ndataset[row][col] == None:
                        # Get neighbors data if current column value is null (None)
                        neighbors = KNN.get_neighbors(ndataset[row], dataset, k_neighbors, col)
                        # Check if the closest neighbor distance is zero
                        if KNN.get_distance(neighbors[0], ndataset[row]) == 0:
                            ndataset[row][col] = neighbors[0][col]
                        else:
                            # Get neighbors column rows data
                            neighbors_col_values = [tmp_row[col] for tmp_row in neighbors]
                            # Process neighbors column rows data
                            if mode == "mean":
                                neighbors_data = statistics.mean(neighbors_col_values)
                                if mean_precision: neighbors_data = round(neighbors_data, mean_precision)
                            elif mode == "median":
                                neighbors_data = statistics.median(neighbors_col_values)
                            elif mode == "mode":
                                neighbors_data = statistics.mode(neighbors_col_values)
                            # Fill current column value from neighbors data
                            ndataset[row][col] = neighbors_data
        # Return normalized dataset
        return ndataset

    # ==============================
    # Finally... ＼(＾O＾)／
    # ==============================

    @staticmethod
    # Can also be used as K-fold cross validation (use list slicing)
    # If answer dataset not provided then accuracy won't be shown
    # Use header_list = None and num_rows = 0 to suppress printing
    # This function return classified test dataset
    def predict_class(test_dataset, train_dataset, header_list, k_neighbors = 5, num_rows = 10, answer_dataset = None):
        # Copy list as new object, not as reference
        ntest_dataset = test_dataset[:]
        # Merge dataset for later imputation
        both_dataset = train_dataset + test_dataset
        # Predict class (last column index) of data, resulting classified data
        both_dataset = KNN.imputer(both_dataset, len(both_dataset[0]) - 1, k_neighbors, "mode")
        # Replace unclassified test_dataset with classified test_dataset
        ntest_dataset = both_dataset[-len(ntest_dataset):]
        # Print the first N rows of test_dataset
        KNN.print_dataset(ntest_dataset, header_list, num_rows)
        # Calculate accuracy if answer_dataset is specified
        if answer_dataset:
            # Get class (last colum index) predictions and the right answers
            predictions = [row[-1] for row in ntest_dataset]
            answers = [row[-1] for row in answer_dataset]
            accuracy = []
            for col in range(len(predictions)):
                # 1 point if prediction is right, 0 point if wrong
                if predictions[col] == answers[col]:
                    accuracy.append(1)
                else: accuracy.append(0)
            # Print predictions accuracy
            print(f"{sum(accuracy)} right answer(s) out of {len(accuracy)} data")
            accuracy = sum(accuracy) / len(accuracy) * 100
            print(f"KNN predictions accuracy: {round(accuracy, 2)}%")
        # Return classified test_dataset
        return ntest_dataset
    
    # TODO: Investigate bug causing same accuracy after the first one
    #       Probably because of stray list reference somewhere?
    #       But it seems that each list address are different
    #       Use the "id()" function to test the address
    # def regression_test(test_dataset, train_dataset, answer_dataset, min_neighbors = 3, max_neighbors = 30):
    #     for i in range(min_neighbors, max_neighbors + 1):
    #         # Print to manually select optimal number of neighbors
    #         print(f"Number of neighbors: {i}")
    #         KNN.predict_class(test_dataset, train_dataset, None, i, 0, answer_dataset)
    #         print()