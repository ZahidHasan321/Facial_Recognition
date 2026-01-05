import face_recognition
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ------------------------------------------------------------------
# SETUP: Define your directory paths here
# ------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"

# ------------------------------------------------------------------
# STEP 1: Load and Encode Known Faces
# ------------------------------------------------------------------
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

# Loop through every file in the known folder
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    # Skip hidden files or non-images usually found in folders
    if filename.startswith("."):
        continue

    # Load the image
    image = face_recognition.load_image_file(filepath)
    
    # Get the face encoding (assuming only one face per known image)
    # We take the first face found ([0])
    try:
        encoding = face_recognition.face_encodings(image)[0]
        
        # Use the filename (minus extension) as the person's name
        name = os.path.splitext(filename)[0]
        
        known_face_encodings.append(encoding)
        known_face_names.append(name)
        print(f" -> Learned: {name}")
        
    except IndexError:
        print(f" -> WARNING: No face found in {filename}. Skipping.")

print(f"\nProcessing {len(known_face_names)} known faces finished.")

# ------------------------------------------------------------------
# STEP 2: Recognize Faces in Unknown Images
# ------------------------------------------------------------------
print("\nChecking unknown images...")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    filepath = os.path.join(UNKNOWN_FACES_DIR, filename)

    if filename.startswith("."):
        continue

    print(f"\nAnalyzing {filename}...")

    # Load unknown image
    unknown_image = face_recognition.load_image_file(filepath)
    pil_image = Image.fromarray(unknown_image)

    # Find all faces and encodings in the unknown image
    unknown_face_locations = face_recognition.face_locations(unknown_image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

    if not unknown_face_encodings:
        print(" - No faces found in this image.")
        continue

    # Store matches for visualization
    matched_faces = []

    # Draw on a copy of the image
    draw = ImageDraw.Draw(pil_image)

    # Check each face found in the unknown image
    for i, (face_encoding, face_location) in enumerate(zip(unknown_face_encodings, unknown_face_locations)):

        # Compare against known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        # If a match was found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            matched_faces.append({
                'name': name,
                'path': os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")  # Try common extensions
            })
            print(f" - Found: {name}!")
        else:
            matched_faces.append({'name': 'Unknown', 'path': None})
            print(" - Found: Unknown Person")

        # Draw bounding box (face_location is in format: top, right, bottom, left)
        top, right, bottom, left = face_location
        draw.rectangle([(left, top), (right, bottom)], outline="green", width=3)

        # Draw label
        draw.rectangle([(left, bottom), (right, bottom + 30)], fill="green")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((left + 6, bottom + 5), name, fill="white", font=font)

    # ------------------------------------------------------------------
    # VISUALIZATION: Create side-by-side display
    # ------------------------------------------------------------------
    num_matches = len(matched_faces)

    # Calculate grid size for matched faces
    grid_cols = min(2, num_matches)
    grid_rows = (num_matches + grid_cols - 1) // grid_cols

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # Left side: Unknown image with annotations
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(pil_image)
    ax1.set_title(f"Unknown Image: {filename}", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right side: Grid of matched known faces
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    ax2.set_title(f"Matched Known Faces ({num_matches} detected)", fontsize=14, fontweight='bold')

    # Create grid for known faces
    for idx, match in enumerate(matched_faces):
        row = idx // grid_cols
        col = idx % grid_cols

        # Create subplot for this face
        inner_ax = plt.subplot(1, 2, 2)
        gs = inner_ax.get_gridspec()
        inner_ax.remove()

        # Calculate position
        subfig_height = 0.8 / grid_rows
        subfig_width = 0.4 / grid_cols
        left_pos = 0.55 + col * subfig_width
        bottom_pos = 0.9 - (row + 1) * subfig_height

        face_ax = fig.add_axes([left_pos, bottom_pos, subfig_width * 0.9, subfig_height * 0.9])

        if match['path'] and os.path.exists(match['path']):
            known_img = Image.open(match['path'])
            face_ax.imshow(known_img)
            face_ax.set_title(match['name'], fontsize=12, fontweight='bold', color='green')
        else:
            # Try different extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = os.path.join(KNOWN_FACES_DIR, f"{match['name']}{ext}")
                if os.path.exists(potential_path):
                    known_img = Image.open(potential_path)
                    face_ax.imshow(known_img)
                    face_ax.set_title(match['name'], fontsize=12, fontweight='bold', color='green')
                    found = True
                    break

            if not found:
                # Show placeholder for unknown
                face_ax.text(0.5, 0.5, '?', ha='center', va='center',
                           fontsize=60, color='red', fontweight='bold')
                face_ax.set_title(match['name'], fontsize=12, fontweight='bold', color='red')

        face_ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally save the figure
    output_path = f"output_{filename}"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" - Saved visualization to: {output_path}")
