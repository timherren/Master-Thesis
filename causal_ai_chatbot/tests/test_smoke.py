"""
Smoke tests for basic application availability.
"""


def test_homepage_smoke(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
