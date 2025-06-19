# main.py
import io
import os
import base64
import logging
import requests
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import uvicorn
from fastapi.responses import FileResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MRI Image Enhancement API", description="API for enhancing MRI images using U-Net")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive file ID and download URL
GOOGLE_DRIVE_FILE_ID = "1CDIpo51ryovIq6-5Eumz6oVJQNXTB11U"
CHECKPOINT_PATH = "generator_checkpoint.pth"

# Global variables for model management
MODEL = None
MODEL_LOADING = False
MODEL_LOADED = False

def download_from_google_drive(file_id: str, destination: str):
    """Download file from Google Drive using gdown library for better reliability"""
    try:
        logger.info(f"Downloading checkpoint from Google Drive...")
        
        # Try using gdown first (more reliable for large files)
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=False, fuzzy=True)
            
            # Verify the downloaded file is a valid PyTorch checkpoint
            checkpoint_test = torch.load(destination, map_location='cpu', weights_only=False)
            if isinstance(checkpoint_test, dict):
                logger.info(f"Checkpoint downloaded and verified successfully using gdown")
                return True
            else:
                logger.error("Downloaded file is not a valid PyTorch checkpoint")
                if os.path.exists(destination):
                    os.remove(destination)
                return False
                
        except ImportError:
            logger.warning("gdown not installed, trying manual method...")
            # Fall back to manual method
            pass
        except Exception as e:
            logger.warning(f"gdown failed: {str(e)}, trying manual method...")
        
        # Manual method as fallback
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        
        session = requests.Session()
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # First request to get any confirmation tokens
        response = session.get(url, headers=headers, stream=True)
        
        # Look for confirmation token in the response
        if 'text/html' in response.headers.get('content-type', ''):
            # Try to extract confirmation token from HTML
            content = response.text
            if 'confirm=' in content:
                import re
                confirm_match = re.search(r'confirm=([^&\s"\']+)', content)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                    response = session.get(url, headers=headers, stream=True)
        
        # Check if we're still getting HTML
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            logger.error("Still receiving HTML. Please ensure the Google Drive file is publicly accessible and try using gdown library.")
            return False
        
        # Save the file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify the downloaded file
        try:
            checkpoint_test = torch.load(destination, map_location='cpu', weights_only=False)
            if isinstance(checkpoint_test, dict):
                logger.info(f"Checkpoint downloaded and verified successfully using manual method")
                return True
            else:
                logger.error("Downloaded file is not a valid PyTorch checkpoint")
                if os.path.exists(destination):
                    os.remove(destination)
                return False
        except Exception as e:
            logger.error(f"Downloaded file is not a valid PyTorch checkpoint: {str(e)}")
            if os.path.exists(destination):
                os.remove(destination)
            return False
        
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        return False

# Helper functions
def cropped_t(x, y):
    delta = (x.shape[-1] - y.shape[-1]) // 2
    return x[:, :, delta:x.shape[-1]-delta, delta:x.shape[-1]-delta]

def d_conv(in_c, out_c, k_size=3):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k_size, padding="same", bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=k_size, padding="same", bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

# Define U-Net model
class GeneratorUnet(nn.Module):
    def __init__(self):
        super(GeneratorUnet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = d_conv(in_c=1, out_c=64, k_size=3)
        self.down2 = d_conv(in_c=64, out_c=128, k_size=3)
        self.down3 = d_conv(in_c=128, out_c=256, k_size=3)
        self.down4 = d_conv(in_c=256, out_c=512, k_size=3)
        self.down5 = d_conv(in_c=512, out_c=1024, k_size=3)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = d_conv(1024, 512, k_size=3)
        self.upconv2 = d_conv(512, 256, k_size=3)
        self.upconv3 = d_conv(256, 128, k_size=3)
        self.upconv4 = d_conv(128, 64, k_size=3)
        self.out_img = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.down1(x)    
        x2 = self.pool(x1)
        x3 = self.down2(x2)   
        x4 = self.pool(x3)
        x5 = self.down3(x4)   
        x6 = self.pool(x5)
        x7 = self.down4(x6)   
        x8 = self.pool(x7)
        x9 = self.down5(x8)
        x10 = self.up1(x9)
        x7c = cropped_t(x7, x10)
        x11 = torch.cat((x7c, x10), dim=1)
        x12 = self.upconv1(x11)
        x13 = self.up2(x12)
        x5c = cropped_t(x5, x13)
        x14 = torch.cat((x5c, x13), dim=1)
        x15 = self.upconv2(x14)
        x16 = self.up3(x15)
        x3c = cropped_t(x3, x16)
        x17 = torch.cat((x3c, x16), dim=1)
        x18 = self.upconv3(x17)
        x19 = self.up4(x18)
        x1c = cropped_t(x1, x19)
        x20 = torch.cat((x1c, x19), dim=1)
        x21 = self.upconv4(x20)
        output_img = self.out_img(x21)
        return output_img

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model function with protection against multiple loading
def load_model():
    global MODEL, MODEL_LOADING, MODEL_LOADED
    
    # Check if model is already loaded
    if MODEL_LOADED and MODEL is not None:
        logger.info("Model already loaded, skipping...")
        return MODEL
    
    # Check if model is currently being loaded
    if MODEL_LOADING:
        logger.info("Model is currently being loaded, waiting...")
        import time
        timeout = 300  # 5 minutes timeout
        elapsed = 0
        while MODEL_LOADING and elapsed < timeout:
            time.sleep(1)
            elapsed += 1
        
        if MODEL_LOADED and MODEL is not None:
            return MODEL
        else:
            logger.error("Model loading timed out or failed")
            return None
    
    # Set loading flag
    MODEL_LOADING = True
    
    try:
        logger.info("Starting model loading process...")
        model = GeneratorUnet().to(device)

        # Check if checkpoint already exists
        if not os.path.exists(CHECKPOINT_PATH):
            logger.info("Checkpoint not found, downloading from Google Drive...")
            success = download_from_google_drive(GOOGLE_DRIVE_FILE_ID, CHECKPOINT_PATH)
            if not success:
                logger.error("Failed to download checkpoint from Google Drive")
                MODEL_LOADING = False
                return None
        else:
            logger.info("Checkpoint already exists, using cached version")

        # Load checkpoint with weights_only=False for compatibility
        logger.info("Loading checkpoint into model...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

        # Load the checkpoint directly as state_dict
        model.load_state_dict(checkpoint)
        logger.info(f"Model loaded successfully from {CHECKPOINT_PATH} on {device}")

        model.eval()
        
        # Update global variables
        MODEL = model
        MODEL_LOADED = True
        MODEL_LOADING = False
        
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        MODEL_LOADING = False
        return None

# Define image preprocessing
preprocess_image = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Helper functions for processing
def read_npy(file_bytes: bytes):
    """Load .npy file from uploaded bytes"""
    try:
        npy_data = np.load(io.BytesIO(file_bytes))
        return npy_data
    except Exception as e:
        logger.error(f"Error reading .npy file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid .npy file or corrupted content"
        )

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded string"""
    try:
        if img_array.dtype != np.uint8:
            # Normalize and convert to uint8
            if img_array.max() > 1.0:
                img_array = img_array.astype(np.float32) / img_array.max()
            img_array = (img_array * 255).astype(np.uint8)
            
        if len(img_array.shape) == 2:  # Handle grayscale images
            image = Image.fromarray(img_array, mode='L')
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Handle single channel images
            image = Image.fromarray(img_array.squeeze(), mode='L')
        else:
            image = Image.fromarray(img_array)
            
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting to base64: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing the image"
        )

# Lazy loading function for API endpoints
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL

# Routes
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup with protection against duplicate loading"""
    logger.info("FastAPI startup event triggered")
    
    # Only attempt to load if not already loaded or loading
    if not MODEL_LOADED and not MODEL_LOADING:
        load_model()
    else:
        logger.info("Model loading skipped - already loaded or in progress")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as file:
        return file.read()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/health")
async def health_check():
    """Health check endpoint that doesn't trigger model loading"""
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """Enhance MRI .npy image using the loaded model"""
    # Get model with lazy loading
    model = get_model()
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please try again later."
        )
    
    try:
        # Read the uploaded file
        file_bytes = await file.read()
        
        # Process .npy file
        if file.filename.endswith('.npy'):
            input_image = read_npy(file_bytes)
            logger.info(f"Original numpy array shape: {input_image.shape}")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .npy files are supported"
            )
        
        # Add padding to match the expected dimensions
        # If we're getting a dimension mismatch error, we need to pad the input
        if len(input_image.shape) == 2:
            from scipy.ndimage import zoom

                # Resize input image to 256x256 if needed
            if input_image.shape[0] != 256 or input_image.shape[1] != 256:
                zoom_factors = (256/input_image.shape[0], 256/input_image.shape[1])
                input_image = zoom(input_image, zoom_factors, order=1)

            input_tensor = torch.from_numpy(input_image).float().unsqueeze(0).unsqueeze(0)  # (batch=1, channel=1, height, width)
            logger.info(f"Processed tensor shape: {input_tensor.shape}")
            input_tensor = input_tensor.to(device)

        else:
            # For other dimensional inputs, try to adapt
            logger.warning(f"Input has unexpected shape: {input_image.shape}. Attempting to adapt.")
            input_tensor = torch.from_numpy(input_image).float()
            
            # Ensure we have 4 dimensions (batch, channel, height, width)
            while len(input_tensor.shape) < 4:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Ensure the channel dimension is 5 as expected
            if input_tensor.shape[1] != 5:
                if input_tensor.shape[1] == 1:
                    input_tensor = input_tensor.repeat(1, 5, 1, 1)
                else:
                    # Other channel counts will be handled case by case
                    if input_tensor.shape[1] < 5:
                        # Pad with zeros
                        padding = torch.zeros(input_tensor.shape[0], 5 - input_tensor.shape[1], 
                                             input_tensor.shape[2], input_tensor.shape[3])
                        input_tensor = torch.cat([input_tensor, padding], dim=1)
                    else:
                        # Truncate
                        input_tensor = input_tensor[:, :5, :, :]
            
            logger.info(f"Adapted tensor shape: {input_tensor.shape}")
            input_tensor = input_tensor.to(device)
        
        # Display the expected input shape for troubleshooting
        logger.info(f"Final input tensor shape: {input_tensor.shape}")
            
        # Generate enhanced image
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        
        logger.info(f"Enhanced tensor shape: {enhanced_tensor.shape}")
        
        # Convert back to numpy array
        enhanced_image = enhanced_tensor.cpu().numpy().squeeze()
        
        # For display, use the first channel if we had to create 5 channels
        if len(input_tensor.shape) >= 3 and input_tensor.shape[1] == 5:
            input_image_for_display = input_tensor[:, 0, :, :].cpu().numpy().squeeze()
        else:
            input_image_for_display = input_tensor.cpu().numpy().squeeze()
        
        # Convert images to base64 for display
        original_base64 = numpy_to_base64(input_image_for_display)
        enhanced_base64 = numpy_to_base64(enhanced_image)
        
        return {
            "status": "success",
            "original_image": original_base64,
            "enhanced_image": enhanced_base64
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Updated for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)