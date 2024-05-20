import csv
import random
from tqdm import tqdm

# Function to slightly modify a prompt
def modify_prompt(prompt):
    # Split prompt into words
    words = prompt.split()
    # Randomly choose one word to replace with a synonym (if possible)
    idx = random.randint(0, len(words)-1)
    # Some simple synonym replacements for demonstration purposes
    synonyms = {
        "detect": ["identify", "spot", "find"],
        "objects": ["entities", "items", "elements"],
        "image": ["picture", "photo", "photograph"],
        "sentiment": ["opinion", "attitude", "emotion"],
        "document": ["text", "article", "paper"],
        "classify": ["categorize", "group", "sort"]
    }
    if words[idx].lower() in synonyms:
        words[idx] = random.choice(synonyms[words[idx].lower()])
    # Join modified words back into a prompt
    modified_prompt = ' '.join(words)
    return modified_prompt

# Load existing prompts from CSV file
existing_prompts = []
with open("C:/Users/omkar/OneDrive/Desktop/prompt data set/training.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        existing_prompts.append(row['Prompt Text'])

# Generate additional prompts
generated_prompts = []
total_prompts = len(existing_prompts) * 1000
with tqdm(total=total_prompts, desc="Generating Prompts") as pbar:
    for prompt in existing_prompts:
        # Generate three modified versions of the prompt
        for _ in range(1000):
            modified = modify_prompt(prompt)
            # Ensure modified prompt is not a duplicate of existing prompts
            while modified in generated_prompts:
                modified = modify_prompt(prompt)
            generated_prompts.append(modified)
            pbar.update(1)

# Write generated prompts to CSV file
with open('larger_dataset.csv', 'w', newline='') as csvfile:
    fieldnames = ['Prompt Text', 'Processing Type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for prompt in generated_prompts:
        writer.writerow({'Prompt Text': prompt, 'Processing Type': 'CV' if 'image' in prompt.lower() else 'Text'})

print("Larger dataset generated successfully.")
