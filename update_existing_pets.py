#!/usr/bin/env python3
"""
Script to update existing pets in the database with missing animal_type field.
This ensures that lost pet scanning will work with existing pets.
"""

import firebase_admin
from firebase_admin import credentials, firestore
import os

def update_existing_pets():
    """Update existing pets with animal_type field if missing"""
    
    # Initialize Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase-credentials.json')
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    try:
        # Get all pets from my_pets collection
        pets_ref = db.collection('my_pets')
        pets = pets_ref.stream()
        
        updated_count = 0
        total_count = 0
        
        for pet_doc in pets:
            total_count += 1
            pet_data = pet_doc.to_dict()
            
            # Check if animal_type is missing
            if 'animal_type' not in pet_data or pet_data['animal_type'] == 'unknown':
                # Try to extract from muzzle_features
                animal_type = 'unknown'
                if 'muzzle_features' in pet_data and pet_data['muzzle_features']:
                    muzzle_features = pet_data['muzzle_features']
                    if isinstance(muzzle_features, list) and len(muzzle_features) > 0:
                        animal_type = muzzle_features[0].get('animal_type', 'unknown')
                    elif isinstance(muzzle_features, dict):
                        animal_type = muzzle_features.get('animal_type', 'unknown')
                
                # Update the pet document
                if animal_type != 'unknown':
                    pet_doc.reference.update({
                        'animal_type': animal_type
                    })
                    updated_count += 1
                    print(f"Updated pet '{pet_data.get('name', 'Unknown')}' with animal_type: {animal_type}")
        
        print(f"\nUpdate complete!")
        print(f"Total pets processed: {total_count}")
        print(f"Pets updated: {updated_count}")
        
    except Exception as e:
        print(f"Error updating pets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_existing_pets()

