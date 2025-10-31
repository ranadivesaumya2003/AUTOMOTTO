// Replace mock API calls with real endpoints
const API_BASE_URL = 'https://your-api.railway.app/api';

class DatabaseService {
  static async getVehicles(userToken) {
    const response = await fetch(`${API_BASE_URL}/vehicles`, {
      headers: {
        'Authorization': `Bearer ${userToken}`,
        'Content-Type': 'application/json'
      }
    });
    return response.json();
  }

  static async getTrips(vehicleId, token) {
    const response = await fetch(`${API_BASE_URL}/trips?vehicle_id=${vehicleId}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.json();
  }

  static async triggerSOS(vehicleId, lat, lng, token) {
    const response = await fetch(`${API_BASE_URL}/sos`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ vehicle_id: vehicleId, lat, lng })
    });
    return response.json();
  }
}
