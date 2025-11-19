import requests
import time
import json
from datetime import datetime, timedelta

# Konfigurasi Supabase
SUPABASE_URL = "https://djdcaivfottxtqvekoud.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRqZGNhaXZmb3R0eHRxdmVrb3VkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1MzY1NDIsImV4cCI6MjA3NjExMjU0Mn0.z-T_fi2vhI8rY79hVkmvJwfK6kdCYj41ChAZjkHBu1Q"
BUCKET_NAME = "camera-captures"

def get_pending_images():
    """Mendapatkan gambar yang statusnya pending atau stuck processing"""
    url = f"{SUPABASE_URL}/rest/v1/yolo_processing"
    
    # Query untuk mendapatkan pending dan processing yang stuck (lebih dari 5 menit)
    params = {
        "select": "*",
        "or": f"(status.eq.pending,and(status.eq.processing,updated_at.lt.{datetime.utcnow() - timedelta(minutes=5)}.toISOString()))",
        "order": "created_at.asc"
    }
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting pending images: {response.status_code} - {response.text}")
        return []

def update_status(filename, status, detected_objects=None, processing_time=None, error_message=None):
    """Update status processing di database"""
    url = f"{SUPABASE_URL}/rest/v1/yolo_processing"
    params = {"filename": f"eq.{filename}"}
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }
    
    if detected_objects is not None:
        update_data["detected_objects"] = detected_objects
    
    if processing_time is not None:
        update_data["processing_time"] = processing_time
        
    if error_message is not None:
        update_data["error_message"] = error_message
        
    if status == "completed":
        update_data["processed_at"] = datetime.utcnow().isoformat() + "Z"
    
    response = requests.patch(url, json=update_data, headers=headers, params=params)
    
    if response.status_code == 200:
        print(f"✓ Updated {filename} to {status}")
        return True
    else:
        print(f"✗ Failed to update {filename}: {response.status_code} - {response.text}")
        return False

def process_with_yolo(image_url):
    """
    Process image dengan YOLO model
    Ganti dengan implementasi YOLO yang sebenarnya
    """
    print(f"Processing image: {image_url}")
    
    # Simulasi processing delay
    time.sleep(2)
    
    # Mock detection results - GANTI DENGAN YOLO YANG SEBENARNYA
    detected_objects = [
        {
            "class": "person",
            "confidence": 0.89,
            "bbox": [100, 150, 200, 300]
        },
        {
            "class": "chair", 
            "confidence": 0.75,
            "bbox": [300, 200, 150, 180]
        }
    ]
    
    return detected_objects, 2.1

def main():
    print("YOLO Processor Started...")
    
    while True:
        try:
            # Dapatkan gambar yang perlu diproses
            pending_images = get_pending_images()
            
            if pending_images:
                print(f"Found {len(pending_images)} images to process")
                
                for image in pending_images:
                    filename = image["filename"]
                    image_url = image["image_url"]
                    current_status = image["status"]
                    
                    print(f"Processing: {filename} (status: {current_status})")
                    
                    # Update status to processing
                    if not update_status(filename, "processing"):
                        continue
                    
                    try:
                        # Process dengan YOLO
                        start_time = time.time()
                        detected_objects, processing_time = process_with_yolo(image_url)
                        
                        # Update status completed
                        update_status(
                            filename, 
                            "completed", 
                            detected_objects, 
                            processing_time
                        )
                        
                        print(f"✓ Completed processing {filename} in {processing_time}s")
                        
                    except Exception as e:
                        error_msg = f"YOLO processing failed: {str(e)}"
                        print(f"✗ {error_msg}")
                        update_status(
                            filename, 
                            "failed", 
                            error_message=error_msg
                        )
                        
            else:
                print("No pending images to process")
                
            # Tunggu sebelum check berikutnya
            time.sleep(10)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
