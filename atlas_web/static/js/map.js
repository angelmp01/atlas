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
let locations = []; // Store all locations for autocomplete
let infoControl = null; // Info panel for hover

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
    
    // Set default date to today
    document.getElementById('date').valueAsDate = new Date();
    
    // Initialize buffer slider
    initBufferSlider();
    
    // Initialize autocomplete fields
    initAutocomplete('origin');
    initAutocomplete('destination');
    
    // Initialize clear buttons
    initClearButtons();
    
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
    
    // Add info control (bottom-right corner)
    infoControl = L.control({position: 'bottomright'});
    
    infoControl.onAdd = function(map) {
        this._div = L.DomUtil.create('div', 'map-info');
        this.update();
        return this._div;
    };
    
    infoControl.update = function(data) {
        if (!data) {
            this._div.innerHTML = '<h4>Información</h4>Pasa el cursor sobre una localización';
        } else if (data.type === 'route') {
            // Show route information
            this._div.innerHTML = `
                <h4>🗺️ Ruta Calculada</h4>
                <div style="line-height: 1.6;">
                    <strong>Origen:</strong> ${data.origin}<br/>
                    <strong>Destino:</strong> ${data.destination}<br/>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                    <strong>Distancia:</strong> ${data.distance} km<br/>
                    <strong>Tiempo:</strong> ${data.time}<br/>
                    <strong>Segmentos:</strong> ${data.segments}
                </div>
            `;
        } else {
            // Show location information (on hover)
            this._div.innerHTML = '<h4>Información</h4><b>' + data.name + '</b><br/>ID: ' + data.id;
        }
    };
    
    infoControl.addTo(map);
    
    console.log('Map initialized');
}

/**
 * Initialize clear buttons for origin and destination
 */
function initClearButtons() {
    const clearOrigin = document.getElementById('clear-origin');
    const clearDestination = document.getElementById('clear-destination');
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    
    clearOrigin.addEventListener('click', () => {
        originInput.value = '';
        delete originInput.dataset.locationId;
        
        // If destination is still selected, show all markers and keep destination highlighted
        const destinationId = destinationInput.dataset.locationId;
        showAllMarkers();
        
        if (destinationId && markers[destinationId]) {
            markers[destinationId].setStyle({
                radius: 8,
                fillColor: '#e74c3c',
                color: '#c0392b',
                weight: 2,
                fillOpacity: 0.9,
                opacity: 1
            });
        }
    });
    
    clearDestination.addEventListener('click', () => {
        destinationInput.value = '';
        delete destinationInput.dataset.locationId;
        
        // If origin is still selected, show all markers and keep origin highlighted
        const originId = originInput.dataset.locationId;
        showAllMarkers();
        
        if (originId && markers[originId]) {
            markers[originId].setStyle({
                radius: 8,
                fillColor: '#27ae60',
                color: '#229954',
                weight: 2,
                fillOpacity: 0.9,
                opacity: 1
            });
        }
    });
}

/**
 * Show all markers on the map
 */
function showAllMarkers() {
    Object.values(markers).forEach(circle => {
        circle.setStyle({ 
            radius: 4,
            fillColor: '#3498db',
            color: '#2980b9',
            weight: 1,
            opacity: 0.8, 
            fillOpacity: 0.6 
        });
    });
    
    // Remove route line if exists
    if (window.routeLine) {
        map.removeLayer(window.routeLine);
        window.routeLine = null;
    }
}

/**
 * Hide all markers except specified ones
 */
function hideMarkersExcept(exceptIds = []) {
    Object.entries(markers).forEach(([id, circle]) => {
        if (exceptIds.includes(id)) {
            circle.setStyle({ opacity: 1, fillOpacity: 0.9 });
        } else {
            circle.setStyle({ opacity: 0, fillOpacity: 0 });
        }
    });
}

/**
 * Update route display when both origin and destination are selected
 */
function updateRouteDisplay() {
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    
    const originId = originInput.dataset.locationId;
    const destinationId = destinationInput.dataset.locationId;
    
    // Only proceed if both are selected
    if (!originId || !destinationId) {
        return;
    }
    
    // Hide all markers except origin and destination
    hideMarkersExcept([originId, destinationId]);
    
    // Highlight origin in green
    if (markers[originId]) {
        markers[originId].setStyle({
            radius: 8,
            fillColor: '#27ae60',
            color: '#229954',
            weight: 2,
            fillOpacity: 0.9,
            opacity: 1
        });
    }
    
    // Highlight destination in red
    if (markers[destinationId]) {
        markers[destinationId].setStyle({
            radius: 8,
            fillColor: '#e74c3c',
            color: '#c0392b',
            weight: 2,
            fillOpacity: 0.9,
            opacity: 1
        });
    }
    
    // Draw route from API
    drawRouteFromAPI(originId, destinationId);
}

/**
 * Select location in form field (origin or destination)
 */
function selectLocationOnMap(location) {
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    
    // If origin is not set, set it
    if (!originInput.dataset.locationId) {
        originInput.value = location.name;
        originInput.dataset.locationId = location.id;
        
        // Keep all markers visible, just highlight origin in green
        if (markers[location.id]) {
            markers[location.id].setStyle({
                radius: 8,
                fillColor: '#27ae60',
                color: '#229954',
                weight: 2,
                fillOpacity: 0.9,
                opacity: 1
            });
        }
        
        showStatus('✅ Origen seleccionado. Ahora selecciona el destino.', 'success', 3000);
        
        // Check if destination is already set
        updateRouteDisplay();
    }
    // If origin is set but destination is not, set destination
    else if (!destinationInput.dataset.locationId) {
        // Don't allow same location
        if (location.id === originInput.dataset.locationId) {
            showStatus('⚠️ El destino debe ser diferente al origen', 'warning', 3000);
            return;
        }
        
        destinationInput.value = location.name;
        destinationInput.dataset.locationId = location.id;
        
        showStatus('✅ Destino seleccionado. Puedes calcular la ruta.', 'success', 3000);
        
        // Update route display (hide markers, draw line, zoom)
        updateRouteDisplay();
    }
    // Both are set, do nothing or show message
    else {
        showStatus('ℹ️ Origen y destino ya están seleccionados. Usa el icono de basura para cambiar.', 'info', 3000);
    }
}

/**
 * Initialize autocomplete for a field
 */
function initAutocomplete(fieldId) {
    const input = document.getElementById(fieldId);
    const suggestionsDiv = document.getElementById(`${fieldId}-suggestions`);
    let selectedIndex = -1;
    let currentSuggestions = [];
    
    // Handle input changes
    input.addEventListener('input', (e) => {
        const query = e.target.value.trim().toLowerCase();
        selectedIndex = -1;
        
        if (query.length === 0) {
            suggestionsDiv.classList.remove('active');
            return;
        }
        
        // Filter locations
        currentSuggestions = locations.filter(loc => 
            loc.name.toLowerCase().includes(query)
        );
        
        // Display suggestions
        if (currentSuggestions.length > 0) {
            suggestionsDiv.innerHTML = currentSuggestions.map((loc, index) => {
                const highlightedName = highlightMatch(loc.name, query);
                return `<div class="autocomplete-suggestion" data-index="${index}" data-id="${loc.id}" data-name="${loc.name}">
                    ${highlightedName}
                </div>`;
            }).join('');
            suggestionsDiv.classList.add('active');
            
            // Add click handlers
            suggestionsDiv.querySelectorAll('.autocomplete-suggestion').forEach(div => {
                div.addEventListener('click', () => {
                    selectSuggestion(input, suggestionsDiv, div.dataset.id, div.dataset.name);
                });
            });
        } else {
            suggestionsDiv.innerHTML = '<div class="autocomplete-no-results">No se encontraron resultados</div>';
            suggestionsDiv.classList.add('active');
        }
    });
    
    // Handle keyboard navigation
    input.addEventListener('keydown', (e) => {
        const suggestions = suggestionsDiv.querySelectorAll('.autocomplete-suggestion');
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectedIndex = Math.min(selectedIndex + 1, suggestions.length - 1);
            updateSelectedSuggestion(suggestions, selectedIndex);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectedIndex = Math.max(selectedIndex - 1, -1);
            updateSelectedSuggestion(suggestions, selectedIndex);
        } else if (e.key === 'Enter' && selectedIndex >= 0) {
            e.preventDefault();
            const selected = suggestions[selectedIndex];
            if (selected) {
                selectSuggestion(input, suggestionsDiv, selected.dataset.id, selected.dataset.name);
            }
        } else if (e.key === 'Escape') {
            suggestionsDiv.classList.remove('active');
        }
    });
    
    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !suggestionsDiv.contains(e.target)) {
            suggestionsDiv.classList.remove('active');
        }
    });
    
    // Focus shows suggestions if there's text
    input.addEventListener('focus', () => {
        if (input.value.trim().length > 0) {
            input.dispatchEvent(new Event('input'));
        }
    });
}

/**
 * Highlight matching text in suggestion
 */
function highlightMatch(text, query) {
    const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<strong>$1</strong>');
}

/**
 * Escape special regex characters
 */
function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Update selected suggestion highlight
 */
function updateSelectedSuggestion(suggestions, index) {
    suggestions.forEach((s, i) => {
        if (i === index) {
            s.classList.add('selected');
            s.scrollIntoView({ block: 'nearest' });
        } else {
            s.classList.remove('selected');
        }
    });
}

/**
 * Select a suggestion
 */
function selectSuggestion(input, suggestionsDiv, locationId, locationName) {
    input.value = locationName;
    input.dataset.locationId = locationId;
    suggestionsDiv.classList.remove('active');
    
    // Update map markers visibility
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    
    const originId = originInput.dataset.locationId;
    const destinationId = destinationInput.dataset.locationId;
    
    if (originId && destinationId) {
        // Validate they are different
        if (originId === destinationId) {
            // Revert the change
            input.value = '';
            delete input.dataset.locationId;
            showStatus('⚠️ El destino debe ser diferente al origen', 'warning', 3000);
            return;
        }
        
        // Both selected, use centralized function
        updateRouteDisplay();
        
    } else if (originId) {
        // Only origin selected, keep all markers visible but highlight origin
        showAllMarkers(); // Reset all
        if (markers[originId]) {
            markers[originId].setStyle({
                radius: 8,
                fillColor: '#27ae60',
                color: '#229954',
                weight: 2,
                fillOpacity: 0.9,
                opacity: 1
            });
        }
    } else {
        // Nothing selected, show all
        showAllMarkers();
    }
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
        
        locations = await response.json();
        console.log('Loaded locations:', locations);
        
        // Add circle markers to map
        locations.forEach(location => {
            if (location.latitude && location.longitude) {
                const circle = L.circleMarker([location.latitude, location.longitude], {
                    radius: 4,
                    fillColor: '#3498db',
                    color: '#2980b9',
                    weight: 1,
                    opacity: 0.8,
                    fillOpacity: 0.6
                }).addTo(map);
                
                // Add hover events
                circle.on('mouseover', function(e) {
                    this.setStyle({
                        radius: 6,
                        fillOpacity: 0.9,
                        weight: 2
                    });
                    infoControl.update(location);
                });
                
                circle.on('mouseout', function(e) {
                    // Check if this is a selected marker
                    const originInput = document.getElementById('origin');
                    const destinationInput = document.getElementById('destination');
                    const isSelected = (originInput.dataset.locationId === location.id) || 
                                      (destinationInput.dataset.locationId === location.id);
                    
                    if (!isSelected) {
                        this.setStyle({
                            radius: 4,
                            fillOpacity: 0.6,
                            weight: 1
                        });
                    }
                    infoControl.update();
                });
                
                // Add click event to select location
                circle.on('click', function(e) {
                    selectLocationOnMap(location);
                    L.DomEvent.stopPropagation(e);
                });
                
                markers[location.id] = circle;
            }
        });
        
        showStatus(`✅ ${locations.length} localizaciones cargadas`, 'success', 2000);
        
    } catch (error) {
        console.error('Error loading locations:', error);
        showStatus(`❌ ERROR: ${error.message}`, 'error', 0);
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());
    
    // Get location IDs from data attributes
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    
    const originId = originInput.dataset.locationId;
    const destinationId = destinationInput.dataset.locationId;
    
    console.log('Form data:', data);
    console.log('Origin ID:', originId, 'Destination ID:', destinationId);
    
    // Validate
    if (!originId || !destinationId) {
        showStatus('Por favor, seleccione origen y destino de la lista', 'error');
        return;
    }
    
    if (originId === destinationId) {
        showStatus('Origen y destino deben ser diferentes', 'error');
        return;
    }
    
    // Update data with IDs
    data.origin = originId;
    data.destination = destinationId;
    data.origin_name = originInput.value;
    data.destination_name = destinationInput.value;
    
    // Show loading
    showStatus('Calculando ruta...', 'info');
    const button = event.target.querySelector('button[type="submit"]');
    button.disabled = true;
    button.innerHTML = '<span class="loading"></span> Calculando...';
    
    try {
        // For now, just highlight the selected locations
        highlightRoute(data.origin, data.destination);
        
        showStatus(`Ruta calculada: ${data.origin_name} → ${data.destination_name}`, 'success');
        
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
    Object.values(markers).forEach(circle => {
        circle.setStyle({
            radius: 4,
            fillColor: '#3498db',
            color: '#2980b9',
            weight: 1,
            fillOpacity: 0.6
        });
    });
    
    // Highlight origin (green)
    if (markers[originId]) {
        markers[originId].setStyle({
            radius: 8,
            fillColor: '#27ae60',
            color: '#229954',
            weight: 2,
            fillOpacity: 0.9
        });
        markers[originId].openPopup();
    }
    
    // Highlight destination (red)
    if (markers[destinationId]) {
        markers[destinationId].setStyle({
            radius: 8,
            fillColor: '#e74c3c',
            color: '#c0392b',
            weight: 2,
            fillOpacity: 0.9
        });
    }
    
    // Draw route from API
    if (markers[originId] && markers[destinationId]) {
        drawRouteFromAPI(originId, destinationId);
    }
}

/**
 * Display calculation results
 */
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    // Format truck type
    const truckType = data.truck_type === 'refrigerated' ? '❄ Refrigerado' : '🚚 Normal';
    
    resultsDiv.innerHTML = `
        <h3>Parámetros de Búsqueda</h3>
        <ul>
            <li><strong>Origen:</strong> ${data.origin_name}</li>
            <li><strong>Destino:</strong> ${data.destination_name}</li>
            <li><strong>Tipo de Camión:</strong> ${truckType}</li>
            <li><strong>Buffer:</strong> ${data.buffer} km</li>
            <li><strong>Fecha:</strong> ${data.date}</li>
        </ul>
        <p><em>Nota: La funcionalidad de cálculo de rutas se implementará próximamente.</em></p>
    `;
}

/**
 * Show status notification (top bar)
 * @param {string} message - Message to display
 * @param {string} type - Type: 'info', 'success', 'error', 'warning'
 * @param {number} timeout - Auto-hide timeout in ms (0 = no auto-hide)
 */
function showStatus(message, type = 'info', timeout = null) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status-notification ${type}`;
    
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
 * Hide status notification
 */
function hideStatus() {
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status-notification';
    if (statusDiv.hideTimeout) {
        clearTimeout(statusDiv.hideTimeout);
        statusDiv.hideTimeout = null;
    }
}

/**
 * Draw route from API using pgRouting
 * @param {string} originId - Origin location ID
 * @param {string} destinationId - Destination location ID
 */
async function drawRouteFromAPI(originId, destinationId) {
    try {
        // Show loading status
        showStatus('Calculando ruta...', 'info');
        
        // Remove previous route line if exists
        if (window.routeLine) {
            map.removeLayer(window.routeLine);
            window.routeLine = null;
        }
        
        // Call route API
        const response = await fetch(
            `${API_BASE_URL}/route?origin_id=${originId}&destination_id=${destinationId}`
        );
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error al calcular la ruta');
        }
        
        const routeData = await response.json();
        
        // Extract coordinates from GeoJSON geometries
        const routeCoordinates = [];
        
        for (const segment of routeData.segments) {
            if (segment.geometry) {
                const geom = JSON.parse(segment.geometry);
                
                // GeoJSON uses [lon, lat] format, Leaflet uses [lat, lon]
                if (geom.type === 'LineString') {
                    const coords = geom.coordinates.map(coord => [coord[1], coord[0]]);
                    routeCoordinates.push(...coords);
                } else if (geom.type === 'Point') {
                    routeCoordinates.push([geom.coordinates[1], geom.coordinates[0]]);
                }
            }
        }
        
        if (routeCoordinates.length === 0) {
            throw new Error('No se encontraron coordenadas de ruta');
        }
        
        // Draw route polyline
        window.routeLine = L.polyline(routeCoordinates, {
            color: '#3498db',
            weight: 4,
            opacity: 0.8
        }).addTo(map);
        
        // Add tooltip with time and distance (always visible, like Google Maps)
        const hours = Math.floor(routeData.summary.total_time_minutes / 60);
        const minutes = Math.round(routeData.summary.total_time_minutes % 60);
        const timeText = hours > 0 ? `${hours} h ${minutes} min` : `${minutes} min`;
        
        const tooltipContent = `
            <div style="background: white; padding: 6px 10px; border-radius: 4px; border: 2px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.2); font-weight: 600; text-align: center; line-height: 1.4;">
                <div style="color: #3498db; font-size: 14px;">${timeText}</div>
                <div style="color: #555; font-size: 12px;">${routeData.summary.total_distance_km} km</div>
            </div>
        `;
        
        // Place tooltip at midpoint of route
        const midPoint = Math.floor(routeCoordinates.length / 2);
        const tooltipLatLng = routeCoordinates[midPoint];
        
        window.routeLine.bindTooltip(tooltipContent, {
            permanent: true,
            direction: 'center',
            className: 'route-tooltip',
            offset: [0, 0]
        }).openTooltip();
        
        // Update info control with route details
        infoControl.update({
            type: 'route',
            origin: routeData.origin.name,
            destination: routeData.destination.name,
            distance: routeData.summary.total_distance_km,
            time: timeText,
            segments: routeData.summary.total_segments
        });
        
        // Fit map to show entire route
        map.fitBounds(window.routeLine.getBounds(), { padding: [50, 50] });
        
        // Show success message
        showStatus(
            `Ruta calculada: ${routeData.summary.total_distance_km} km, ${routeData.summary.total_time_minutes} min`,
            'success',
            5000
        );
        
        console.log('Route drawn:', routeData.summary);
        
    } catch (error) {
        console.error('Error drawing route:', error);
        showStatus(`Error al dibujar la ruta: ${error.message}`, 'error', 0);
        
        // Fallback: just fit map to show both markers
        if (markers[originId] && markers[destinationId]) {
            const originLatLng = markers[originId].getLatLng();
            const destLatLng = markers[destinationId].getLatLng();
            const bounds = L.latLngBounds([originLatLng, destLatLng]);
            map.fitBounds(bounds, { padding: [50, 50] });
        }
    }
}
