import requests
from requests.auth import HTTPBasicAuth

# ganti sesuai server Nx kamu
url = "https://192.168.2.226:7001/rest/v3/devices/*/bookmarks"
params = {
    "limit": 5,
    "order": "desc",
    "_orderBy": "startTimeMs"
}

response = requests.get(
    url,
    params=params,
    auth=HTTPBasicAuth("admin", "rangga7671234"),  # ganti sesuai akun Nx
    verify=False  # abaikan SSL warning
)

if response.status_code == 200:
    bookmarks = response.json()
    for bm in bookmarks:
        print(f"{bm['name']} | {bm['startTimeMs']} | {bm['deviceId']}")
else:
    print("Error:", response.status_code, response.text)
