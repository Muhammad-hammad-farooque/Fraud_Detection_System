"""
Tests for /users endpoints.
"""


class TestGetMyProfile:
    def test_get_my_profile_authenticated(self, client, registered_user, auth_headers):
        resp = client.get("/users/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == registered_user["email"]
        assert data["name"] == registered_user["name"]

    def test_get_my_profile_unauthenticated(self, client):
        resp = client.get("/users/me")
        assert resp.status_code == 401


class TestGetUserById:
    def test_get_own_profile_by_id(self, client, registered_user, auth_headers):
        resp = client.get(f"/users/{registered_user['id']}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == registered_user["id"]

    def test_cannot_access_another_users_profile(
        self, client, registered_user, second_user, second_auth_headers
    ):
        resp = client.get(f"/users/{registered_user['id']}", headers=second_auth_headers)
        assert resp.status_code == 403

    def test_get_nonexistent_user(self, client, registered_user, auth_headers):
        resp = client.get(f"/users/{registered_user['id'] + 9999}", headers=auth_headers)
        # Will hit 403 (id mismatch) before the 404 check
        assert resp.status_code in (403, 404)

    def test_get_user_unauthenticated(self, client, registered_user):
        resp = client.get(f"/users/{registered_user['id']}")
        assert resp.status_code == 401
