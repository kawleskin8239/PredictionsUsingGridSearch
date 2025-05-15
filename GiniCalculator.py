# Function to calculate Gini Impurity from a list of class sample sizes
def calc_gini_impurity(sample_sizes):
    total = sum(sample_sizes)
    prob_sum = 0
    for var in sample_sizes:
        prob_sum += (var/total)**2
    return  1 - prob_sum

# Example list of sample sizes
sample_list = [10, 2, 5, 7, 4, 3, 7, 8, 93, 56]

# Print the calculated Gini Impurity
print(f"Gini Impurity of sample list: " + str(calc_gini_impurity(sample_list)))