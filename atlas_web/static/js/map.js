/**
 * ATLAS Map and Form Interaction
 * 
 * Handles:
 * - Loading locations and goods types from API
 * - Form submission and validation
 * - Map initialization and interaction
 */

// API base URL (set by server)
const API_BASE_URL = window.API_BASE_URL || 'http://127.0.0.1:8000';

// Global map instance
let map = null;
let markers = {};

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('ATLAS Web initialized');
    console.log('API Base URL:', API_BASE_URL);
    
    // Initialize map
    initMap();
    
    // Load data from API
    await loadLocations();
    await loadTruckTypes();
    
    // Set default date to today
    document.getElementById('date').valueAsDate = new Date();
    
    // Initialize buffer slider
    initBufferSlider();
    
    // Attach form handler
    document.getElementById('route-form').addEventListener('submit', handleFormSubmit);
});

/**
 * Initialize Leaflet map
 */
function initMap() {
    // Center on Catalonia
    map = L.map('map').setView([41.5, 1.5], 8);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18,
    }).addTo(map);
    
    console.log('Map initialized');
}

/**
 * Initialize buffer range slider
 */
function initBufferSlider() {
    const slider = document.getElementById('buffer');
    const valueDisplay = document.getElementById('buffer-value');
    
    // Update display when slider changes
    slider.addEventListener('input', (e) => {
        valueDisplay.textContent = e.target.value;
        updateSliderBackground(slider);
    });
    
    // Initial background update
    updateSliderBackground(slider);
}

/**
 * Update slider background gradient based on value
 */
function updateSliderBackground(slider) {
    const min = parseInt(slider.min);
    const max = parseInt(slider.max);
    const value = parseInt(slider.value);
    const percentage = ((value - min) / (max - min)) * 100;
    
    slider.style.background = `linear-gradient(to right, var(--secondary-color) 0%, var(--secondary-color) ${percentage}%, var(--border-color) ${percentage}%, var(--border-color) 100%)`;
}

/**
 * Load locations from API and populate select dropdowns
 */
async function loadLocations() {
    try {
        showStatus('Cargando localizaciones...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/locations`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const locations = await response.json();
        console.log('Loaded locations:', locations);
        
        // Populate origin and destination dropdowns
        const originSelect = document.getElementById('origin');
        const destinationSelect = document.getElementById('destination');
        
        originSelect.innerHTML = '<option value="">Seleccione origen...</option>';
        destinationSelect.innerHTML = '<option value="">Seleccione destino...</option>';
        
        locations.forEach(location => {
            const option1 = new Option(location.name, location.id);
            const option2 = new Option(location.name, location.id);
            originSelect.add(option1);
            destinationSelect.add(option2);
            
            // Add marker to map
            if (location.latitude && location.longitude) {
                const marker = L.marker([location.latitude, location.longitude])
                    .bindPopup(`<b>${location.name}</b><br>ID: ${location.id}`)
                    .addTo(map);
                markers[location.id] = marker;
            }
        });
        
        showStatus(`✅ ${locations.length} localizaciones cargadas`, 'success', 2000);
        
    } catch (error) {
        console.error('Error loading locations:', error);
        showStatus(`❌ ERROR: ${error.message}`, 'error', 0);
    }
}

/**
 * Load truck types from API and populate select dropdown
 */
async function loadTruckTypes() {
    try {
        const response = await fetch(`${API_BASE_URL}/goods-types/truck-types`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const truckTypes = await response.json();
        console.log('Loaded truck types:', truckTypes);
        
        const select = document.getElementById('truck-type');
        select.innerHTML = '<option value="">Seleccione tipo...</option>';
        
        truckTypes.forEach(truck => {
            const option = new Option(
                `${truck.name} - ${truck.description}`,
                truck.id
            );
            select.add(option);
        });
        
    } catch (error) {
        console.error('Error loading truck types:', error);
        showStatus(`Error cargando tipos de camión: ${error.message}`, 'error');
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());
    
    console.log('Form data:', data);
    
    // Validate
    if (!data.origin || !data.destination) {
        showStatus('Por favor, seleccione origen y destino', 'error');
        return;
    }
    
    if (data.origin === data.destination) {
        showStatus('Origen y destino deben ser diferentes', 'error');
        return;
    }
    
    // Show loading
    showStatus('Calculando ruta...', 'info');
    const button = event.target.querySelector('button[type="submit"]');
    button.disabled = true;
    button.innerHTML = '<span class="loading"></span> Calculando...';
    
    try {
        // For now, just highlight the selected locations
        highlightRoute(data.origin, data.destination);
        
        showStatus(`Ruta calculada: ${data.origin} → ${data.destination}`, 'success');
        
        // Show results
        displayResults(data);
        
    } catch (error) {
        console.error('Error calculating route:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.innerHTML = '🔍 Calcular Ruta';
    }
}

/**
 * Highlight route on map
 */
function highlightRoute(originId, destinationId) {
    // Reset all markers to default
    Object.values(markers).forEach(marker => {
        marker.setIcon(new L.Icon.Default());
    });
    
    // Highlight origin (green)
    if (markers[originId]) {
        markers[originId].setIcon(L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        }));
        markers[originId].openPopup();
    }
    
    // Highlight destination (red)
    if (markers[destinationId]) {
        markers[destinationId].setIcon(L.icon({
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        }));
    }
    
    // Draw line between origin and destination
    if (markers[originId] && markers[destinationId]) {
        const originLatLng = markers[originId].getLatLng();
        const destLatLng = markers[destinationId].getLatLng();
        
        // Remove previous route line
        if (window.routeLine) {
            map.removeLayer(window.routeLine);
        }
        
        // Draw new route line
        window.routeLine = L.polyline([originLatLng, destLatLng], {
            color: '#3498db',
            weight: 3,
            opacity: 0.7,
            dashArray: '10, 10'
        }).addTo(map);
        
        // Fit map to show both markers
        const bounds = L.latLngBounds([originLatLng, destLatLng]);
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

/**
 * Display calculation results
 */
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    resultsDiv.innerHTML = `
        <h3>Parámetros de Búsqueda</h3>
        <ul>
            <li><strong>Origen:</strong> ${data.origin}</li>
            <li><strong>Destino:</strong> ${data.destination}</li>
            <li><strong>Tipo de Camión:</strong> ${data.truck_type}</li>
            <li><strong>Buffer:</strong> ${data.buffer} km</li>
            <li><strong>Fecha:</strong> ${data.date}</li>
        </ul>
        <p><em>Nota: La funcionalidad de cálculo de rutas se implementará próximamente.</em></p>
    `;
}

/**
 * Show status message
 * @param {string} message - Message to display
 * @param {string} type - Type: 'info', 'success', 'error', 'warning'
 * @param {number} timeout - Auto-hide timeout in ms (0 = no auto-hide)
 */
function showStatus(message, type = 'info', timeout = null) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;
    
    // Clear any existing timeout
    if (statusDiv.hideTimeout) {
        clearTimeout(statusDiv.hideTimeout);
        statusDiv.hideTimeout = null;
    }
    
    // Set auto-hide if timeout specified (and not 0)
    if (timeout !== null && timeout !== 0) {
        statusDiv.hideTimeout = setTimeout(() => hideStatus(), timeout);
    }
}

/**
 * Hide status message
 */
function hideStatus() {
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status-message';
    if (statusDiv.hideTimeout) {
        clearTimeout(statusDiv.hideTimeout);
        statusDiv.hideTimeout = null;
    }
}
