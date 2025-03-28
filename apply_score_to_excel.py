import json
import pandas as pd

# Load the JSON file
with open("requirement_matches.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare a list to hold our rows
rows = []

for item in data:
    requirement = item.get("requirement", "")
    relevance_score = item.get("relevance_score", "")
    matches = item.get("matches", [])
    
    # Build the "Past Performance" cell content
    past_performance_parts = []
    for match in matches:
        document = match.get("document", "")
        sentence = match.get("sentence", "")
        similarity_score = match.get("similarity_score", 0)
        # Round the similarity score to three decimals
        similarity_str = f"{round(similarity_score, 3):.3f}"
        part = f'{document}: "{sentence}"\nScore: {similarity_str}'
        past_performance_parts.append(part)
    
    # Join all matches with two newline characters
    past_performance = "\n\n".join(past_performance_parts)
    
    rows.append({
        "Requirement": requirement,
        "Relevance Score": relevance_score,
        "Past Performance": past_performance
    })

# Create a DataFrame from the rows
df = pd.DataFrame(rows)

# Write the DataFrame to an Excel file
excel_filename = input("Provide an output excel filename (include the .xlsx)\n")
df.to_excel(excel_filename, index=False)

print(f"Excel file '{excel_filename}' created successfully.")
