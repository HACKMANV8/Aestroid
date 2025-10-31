const axios = require('axios');

// Mock location data for testing
const mockLocations = [
  { unitId: 'TANK-001', unitType: 'tank', latitude: 12.9716, longitude: 77.5946 },
  { unitId: 'HELI-002', unitType: 'helicopter', latitude: 13.0827, longitude: 80.2707 },
  { unitId: 'TROOP-003', unitType: 'infantry', latitude: 19.0760, longitude: 72.8777 },
  { unitId: 'DRONE-004', unitType: 'drone', latitude: 28.7041, longitude: 77.1025 },
  { unitId: 'VEHICLE-005', unitType: 'armored_vehicle', latitude: 22.5726, longitude: 88.3639 },
];

async function sendLocationUpdate(location) {
  try {
    const response = await axios.post('http://localhost:5000/api/location', location);
    console.log(`✅ Sent ${location.unitType} location:`, response.data.message);
  } catch (error) {
    console.error(`❌ Failed to send location:`, error.message);
  }
}

async function simulateLocationUpdates() {
  console.log('🚀 Starting location simulation...\n');
  
  for (let i = 0; i < mockLocations.length; i++) {
    const location = mockLocations[i];
    
    // Add some random variation to coordinates
    const randomLat = location.latitude + (Math.random() - 0.5) * 0.01;
    const randomLng = location.longitude + (Math.random() - 0.5) * 0.01;
    
    await sendLocationUpdate({
      ...location,
      latitude: parseFloat(randomLat.toFixed(6)),
      longitude: parseFloat(randomLng.toFixed(6))
    });
    
    // Wait 2 seconds between updates
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  console.log('\n✨ Location simulation completed!');
}

// Run the simulation
simulateLocationUpdates().catch(console.error);