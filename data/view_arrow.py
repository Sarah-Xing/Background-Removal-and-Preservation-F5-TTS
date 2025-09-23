import pyarrow as pa
import pyarrow.ipc as ipc

# Path to the .arrow file
arrow_file_path = 'JLSpeech_char/raw.arrow'

# Open the .arrow file
with pa.memory_map(arrow_file_path, "r") as source:
    reader = ipc.RecordBatchFileReader(source)
    table = reader.read_all()

# Print the table schema
print("Schema:")
print(table.schema)

# Print the first few rows
print("\nFirst few rows:")
print(table.to_pandas().head())
