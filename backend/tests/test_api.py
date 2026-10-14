import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import FastAPI app
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.database.session import init_db

@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_ready" in data
    assert "device" in data

def test_model_info_endpoint(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["backbone_architecture"] == "convnext_tiny"
    assert data["coral_head_type"] == "monotonic_coral"
    assert data["num_grades"] == 5
    assert len(data["grades"]) == 5
    assert "Apple" in data["supported_species"]
    assert "research_evaluation" in data
    assert data["research_evaluation"]["image_only_qwk_mean"] == 0.8143

def test_system_info_endpoint(client):
    response = client.get("/api/v1/system-info")
    assert response.status_code == 200
    data = response.json()
    assert data["application_name"] == "FruitGrade AI"
    assert "pytorch_version" in data
    assert data["max_upload_size_mb"] == 15
    assert ".jpg" in data["allowed_extensions"]

def test_predict_real_image(client):
    # Create or use real test image
    test_img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test_apple.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "analysis_id" in data
    assert "species" in data
    assert "ordinal_grade" in data
    assert "ood" in data
    assert "selective_routing" in data
    assert "inference_latency_ms" in data
    assert data["ordinal_grade"]["grade"] in [0, 1, 2, 3, 4]
    assert data["ordinal_grade"]["grade_name"] in ["Pristine", "Mild", "Moderate", "Severe", "Critical"]
    assert len(data["ordinal_grade"]["discrete_probabilities"]) == 5

def test_predict_unsupported_file_extension(client):
    text_data = io.BytesIO(b"This is a text file, not a fruit image.")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", text_data, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_predict_empty_file(client):
    empty_data = io.BytesIO(b"")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("empty.jpg", empty_data, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]

def test_predict_corrupt_image(client):
    corrupt_data = io.BytesIO(b"\xff\xd8\xff\xe0FAKE_CORRUPTED_BYTES_HERE")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("corrupt.jpg", corrupt_data, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["detail"]

def test_history_crud_flow(client):
    # 1. Query history list
    res_list = client.get("/api/v1/history?limit=10")
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert "total" in data_list
    assert "items" in data_list
    assert len(data_list["items"]) >= 1
    
    first_item = data_list["items"][0]
    analysis_id = first_item["id"]
    
    # 2. Get history detail
    res_detail = client.get(f"/api/v1/history/{analysis_id}")
    assert res_detail.status_code == 200
    data_detail = res_detail.json()
    assert data_detail["id"] == analysis_id
    assert "discrete_probabilities" in data_detail
    
    # 3. Delete single history record
    res_del = client.delete(f"/api/v1/history/{analysis_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"
    
    # Verify it is deleted
    res_check = client.get(f"/api/v1/history/{analysis_id}")
    assert res_check.status_code == 404
    
    # 4. Clear all history with confirmation
    res_clear = client.delete("/api/v1/history?confirm=true")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "success"
    
    # Verify count is now 0
    res_empty = client.get("/api/v1/history")
    assert res_empty.json()["total"] == 0
