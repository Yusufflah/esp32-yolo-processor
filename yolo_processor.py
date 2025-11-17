import os
import uuid
import requests
import cv2
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime
from supabase import create_client

print("🚀 Starting YOLO Image Processor...")

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ SUPABASE_URL or SUPABASE_KEY not set")
    exit(1)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Supabase client initialized")

def download_image(image_url):
    """Download image from Supabase storage"""
    try:
        print(f"📥 Downloading image: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        print(f"✅ Image downloaded: {image.size}")
        return image
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        raise

def process_with_yolo(image):
    """Process image with YOLO model"""
    try:
        print("🔍 Loading YOLO model...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
        # Convert PIL to OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        print("🎯 Running YOLO inference...")
        results = model(opencv_image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Filter low confidence detections
                if confidence < 0.5:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id
                })
                
                # Draw bounding box and label
                cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(opencv_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"✅ YOLO processing completed: {len(detections)} detections")
        
        # Convert back to PIL
        processed_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        return processed_image, detections
        
    except Exception as e:
        print(f"❌ Error in YOLO processing: {e}")
        raise

def upload_processed_image(image, filename):
    """Upload processed image to Supabase"""
    try:
        print(f"📤 Uploading processed image: {filename}")
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Upload to processed-images bucket
        response = supabase.storage.from_("processed-images").upload(
            file=img_byte_arr,
            path=filename,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_("processed-images").get_public_url(filename)
        print(f"✅ Image uploaded: {public_url}")
        return public_url
        
    except Exception as e:
        print(f"❌ Error uploading processed image: {e}")
        raise

def update_processing_status(record_id, status, processed_url=None, detections=None):
    """Update processing status in database"""
    update_data = {
        "status": status,
        "processed_at": datetime.now().isoformat()
    }
    
    if processed_url:
        update_data["processed_image_url"] = processed_url
    if detections is not None:
        update_data["detection_data"] = detections
        
    try:
        supabase.table("yolo_processing").update(update_data).eq("id", record_id).execute()
        print(f"✅ Database updated: {record_id} -> {status}")
    except Exception as e:
        print(f"❌ Error updating database: {e}")

def main():
    """Main processing function"""
    print("\n" + "="*50)
    print("🔄 Checking for pending images...")
    
    try:
        # Get pending images from database
        response = supabase.table("yolo_processing").select("*").eq("status", "pending").execute()
        
        if not response.data:
            print("✅ No pending images to process")
            return
            
        pending_count = len(response.data)
        print(f"📋 Found {pending_count} pending image(s)")
        
        for record in response.data:
            record_id = record["id"]
            original_url = record["original_image_url"]
            
            print(f"\n🖼️ Processing image {record_id}")
            print(f"📷 Original URL: {original_url}")
            
            try:
                # Update status to processing
                update_processing_status(record_id, "processing")
                
                # Download original image
                original_image = download_image(original_url)
                
                # Process with YOLO
                processed_image, detections = process_with_yolo(original_image)
                
                # Generate unique filename
                processed_filename = f"yolo_processed_{uuid.uuid4().hex}.jpg"
                
                # Upload processed image
                processed_url = upload_processed_image(processed_image, processed_filename)
                
                # Update database with results
                update_processing_status(record_id, "completed", processed_url, detections)
                
                print(f"🎉 Successfully processed {record_id}")
                
            except Exception as e:
                print(f"💥 Failed to process {record_id}: {e}")
                update_processing_status(record_id, "failed")
                
    except Exception as e:
        print(f"💥 Error in main process: {e}")

if __name__ == "__main__":
    main()
    print("\n🏁 YOLO Processor finished")
