import os
import requests
import json
import time
from datetime import datetime
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

# Debug: Print environment variables (without revealing full keys)
print("=== Environment Variables Check ===")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

print(f"SUPABASE_URL: {supabase_url}")
if supabase_key:
    print(f"SUPABASE_KEY: {supabase_key[:10]}...")  # Only show first 10 chars
else:
    print("SUPABASE_KEY: NOT SET")

# Check if environment variables are set
if not supabase_url or not supabase_key:
    print("❌ ERROR: Supabase environment variables are not set!")
    print("Please check your GitHub Secrets:")
    print("1. SUPABASE_URL should be your Supabase project URL")
    print("2. SUPABASE_SERVICE_KEY should be your Supabase service_role key (not anon key)")
    exit(1)

print("✅ Environment variables are set correctly")
print("=====================================")

# Load YOLO model
try:
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit(1)

def download_image(image_url):
    """Download image from URL"""
    try:
        print(f"Downloading image: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None

def process_image_with_yolo(image):
    """Process image with YOLO and return annotated image and detection results"""
    try:
        print("Processing image with YOLO...")
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        results = model(opencv_image)
        
        # Annotate image with detections
        annotated_image = results[0].plot()
        
        # Convert back to PIL Image
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        # Extract detection information
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        print(f"✅ YOLO processing completed. Detected {len(detections)} objects")
        return annotated_pil, detections
    except Exception as e:
        print(f"❌ Error processing image with YOLO: {e}")
        return None, []

def upload_processed_image(image, filename):
    """Upload processed image to Supabase storage"""
    try:
        print(f"Uploading processed image: processed_{filename}")
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        
        # Upload to processed-images bucket using REST API
        upload_url = f"{supabase_url}/storage/v1/object/processed-images/processed_{filename}"
        
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "image/jpeg"
        }
        
        response = requests.post(
            upload_url,
            data=img_byte_arr.getvalue(),
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            public_url = f"{supabase_url}/storage/v1/object/public/processed-images/processed_{filename}"
            print(f"✅ Processed image uploaded: {public_url}")
            return public_url
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading processed image: {e}")
        return None

def update_processing_status(record_id, status, processed_image_url=None, processing_result=None, error_message=None, processing_time=None):
    """Update processing status in database"""
    try:
        print(f"Updating status for {record_id} to {status}")
        
        update_data = {
            "status": status,
            "processed": status == "completed",
            "updated_at": datetime.now().isoformat()
        }
        
        if processed_image_url:
            update_data["processed_image_url"] = processed_image_url
            
        if processing_result:
            update_data["processing_result"] = processing_result
            
        if error_message:
            update_data["error_message"] = error_message
            
        if processing_time:
            update_data["processing_time"] = processing_time
            
        if status == "completed":
            update_data["processed_at"] = datetime.now().isoformat()
        
        # Use REST API directly
        url = f"{supabase_url}/rest/v1/yolo_processing?id=eq.{record_id}"
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        response = requests.patch(url, json=update_data, headers=headers, timeout=10)
        
        if response.status_code in [200, 204]:
            print(f"✅ Status updated to {status}")
            return True
        else:
            print(f"❌ Failed to update status: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating processing status: {e}")
        return False

def fetch_pending_images():
    """Fetch pending images from database"""
    try:
        print("Fetching pending images from database...")
        
        url = f"{supabase_url}/rest/v1/yolo_processing?status=eq.pending"
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            records = response.json()
            print(f"✅ Found {len(records)} pending images")
            return records
        else:
            print(f"❌ Failed to fetch pending images: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching pending images: {e}")
        return []

def process_pending_images():
    """Process all pending images in the database"""
    pending_records = fetch_pending_images()
    
    if not pending_records:
        print("No pending images to process")
        return
    
    for record in pending_records:
        record_id = record['id']
        filename = record['filename']
        original_url = record['original_image_url']
        
        print(f"\n🎯 Processing image: {filename}")
        
        # Update status to processing
        if not update_processing_status(record_id, "processing"):
            continue
        
        start_time = time.time()
        
        try:
            # Download original image
            original_image = download_image(original_url)
            if not original_image:
                raise Exception("Failed to download image")
            
            # Process with YOLO
            processed_image, detections = process_image_with_yolo(original_image)
            if not processed_image:
                raise Exception("YOLO processing failed")
            
            # Upload processed image
            processed_url = upload_processed_image(processed_image, filename)
            if not processed_url:
                raise Exception("Failed to upload processed image")
            
            processing_time = time.time() - start_time
            
            # Update record with results
            success = update_processing_status(
                record_id, 
                "completed", 
                processed_url, 
                detections,
                processing_time=processing_time
            )
            
            if success:
                print(f"🎉 Successfully processed {filename} in {processing_time:.2f}s")
                print(f"   Detections: {len(detections)} objects")
            else:
                print(f"⚠️ Image processed but database update failed for {filename}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"❌ Error processing {filename}: {error_msg}")
            update_processing_status(
                record_id, 
                "failed", 
                error_message=error_msg,
                processing_time=processing_time
            )

if __name__ == "__main__":
    print("🚀 Starting YOLO Image Processor...")
    process_pending_images()
    print("✅ Processing completed!")
