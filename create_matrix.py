import pandas as pd

# Load the Excel file and the specific sheet
file_path = 'signal_part_1.xlsx'
sheet_name = 'HV-VH simulation'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract columns
bit = data['bit']
alice_h = data['Alice-H']
alice_v = data['Alice-V']
bob_h = data['Bob-H']
bob_v = data['Bob-V']

# Calculate the max values
max_a_h = alice_h.max()
max_a_v = alice_v.max()
max_b_h = bob_h.max()
max_b_v = bob_v.max()

# Calculate the number of 0s and 1s
num_1 = bit.sum()
num_0 = len(bit) - num_1

# Define helper function to calculate PHH-HH, etc.
def calculate_probability(bit_val, alice_col, bob_col, max_a, max_b, num):
    if num == 0:
        return 0
    mask = (bit == bit_val)
    return (alice_col[mask] * bob_col[mask]).sum() / (max_a * max_b * num)

# Calculate probabilities
PHH_HH = calculate_probability(0, alice_h, bob_h, max_a_h, max_b_h, num_0)
PHH_HV = calculate_probability(0, alice_h, bob_v, max_a_h, max_b_v, num_0)
PHH_VH = calculate_probability(0, alice_v, bob_h, max_a_v, max_b_h, num_0)
PHH_VV = calculate_probability(0, alice_v, bob_v, max_a_v, max_b_v, num_0)

PHV_HH = 0
PHV_HV = 0
PHV_VH = 0
PHV_VV = 0

PVH_HH = 0
PVH_HV = 0
PVH_VH = 0
PVH_VV = 0

PVV_HH = calculate_probability(1, alice_h, bob_h, max_a_h, max_b_h, num_1)
PVV_HV = calculate_probability(1, alice_h, bob_v, max_a_h, max_b_v, num_1)
PVV_VH = calculate_probability(1, alice_v, bob_h, max_a_v, max_b_h, num_1)
PVV_VV = calculate_probability(1, alice_v, bob_v, max_a_v, max_b_v, num_1)

# Define the matrix
prob_matrix = [
    [PHH_HH, PHV_HH, PVH_HH, PVV_HH],
    [PHH_HV, PHV_HV, PVH_HV, PVV_HV],
    [PHH_VH, PHV_VH, PVH_VH, PVV_VH],
    [PHH_VV, PHV_VV, PVH_VV, PVV_VV]
]

# Convert the matrix to a DataFrame
df = pd.DataFrame(prob_matrix, columns=['PHH-HH', 'PHV-HH', 'PVH-HH', 'PVV-HH'])

# Write the DataFrame to a new Excel file
output_file = 'output_probabilities.xlsx'
df.to_excel(output_file, index=False)

print(f"Probabilities matrix has been saved to {output_file}")
