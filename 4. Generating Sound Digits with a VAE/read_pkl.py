import pickle

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    # MIN_MAX_VALUES_PATH = r"D:\personal projects\Music Generation\sound data\MIN_MAX_VALUES_SAVE_DIR\min_max_values.pkl"

    file_path = "D:\personal projects\Music Generation\sound data\MIN_MAX_VALUES_SAVE_DIR\min_max_values.pkl"  # Change this to your file path
    data = read_pickle_file(file_path)
    if data is not None:
        print("Data loaded successfully:")
        print(data)
