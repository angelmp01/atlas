/**
 * ATLAS Map and Form Interaction
 * 
 * Handles:
 * - Loading locations and goods types from API
 * - Form submission and validation
 * - Map initialization and interaction
 */

/**
 * Format location name by removing "agregacion de municipios" and reordering comma-separated parts
 * @param {string} rawName - Raw location name from database
 * @returns {string} - Formatted location name
 */
function formatLocationName(rawName) {
    if (!rawName) return '';
    
    // Remove "agregacion de municipios" (case insensitive)
    let name = rawName.replace(/agregacion de municipios/gi, '').trim();
    
    // Handle comma-separated names (e.g., "Escala, L'" or "Franqueses del Vallès, Les")
    if (name.includes(',')) {
        const parts = name.split(',').map(p => p.trim());
        if (parts.length === 2) {
            const [first, second] = parts;
            // Check if second part ends with apostrophe
            if (second.endsWith("'")) {
                // Concatenate without space: "L'" + "Escala" = "L'Escala"
                name = second + first;
            } else {
                // Concatenate with space: "Les" + " " + "Franqueses del Vallès"
                name = second + ' ' + first;
            }
        }
    }
    
    return name;
}

// Global map instance (API_BASE_URL, MAP_CENTER, MAP_ZOOM are set by server in HTML)
let map = null;
let markers = {};
let locations = []; // Store all locations for autocomplete
let infoControl = null; // Info panel for hover
let layerGroups = {}; // Layer groups for visualization
let currentCandidates = []; // Store current inference candidates for metric visualization

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
    
    // Initialize load slider
    initLoadSlider();
    
    // Initialize autocomplete fields
    initAutocomplete('origin');
    initAutocomplete('destination');
    
    // Initialize clear buttons
    initClearButtons();
    
    // Attach form handler
    document.getElementById('route-form').addEventListener('submit', handleFormSubmit);
    
    // Initialize form validation
    initFormValidation();
    
    // Initialize results panel toggle buttons
    const toggleBtn = document.getElementById('toggle-results-btn');
    const closeBtn = document.getElementById('close-results-btn');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleResultsPanel);
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', closeResultsPanel);
    }
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
                    <strong>Origen:</strong> ${formatLocationName(data.origin)}<br/>
                    <strong>Destino:</strong> ${formatLocationName(data.destination)}<br/>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                    <strong>Distancia:</strong> ${data.distance} km<br/>
                    <strong>Tiempo:</strong> ${data.time}<br/>
                    <strong>Segmentos:</strong> ${data.segments}
                </div>
            `;
        } else {
            // Show location information (on hover)
            this._div.innerHTML = '<h4>Información</h4><b>' + formatLocationName(data.name) + '</b><br/>ID: ' + data.id;
        }
    };
    
    infoControl.addTo(map);
    
    // Initialize layer groups for visualization
    layerGroups = {
        baseRoute: L.layerGroup().addTo(map),
        allCandidates: L.layerGroup().addTo(map),
        feasibleCandidates: L.layerGroup().addTo(map),
        probabilityHeatmap: L.layerGroup().addTo(map),
        priceHeatmap: L.layerGroup().addTo(map),
        weightHeatmap: L.layerGroup().addTo(map),
        scores: L.layerGroup().addTo(map),
        etaHeatmap: L.layerGroup().addTo(map),
        deltaDistance: L.layerGroup().addTo(map),
        alternativeRoutes: L.layerGroup().addTo(map),
        selectedRoute: L.layerGroup().addTo(map)
    };
    
    // Add layer control for new visualization features
    addLayerControl();
    
    console.log('Map initialized');
}

/**
 * Setup event listeners for layer control icons
 */
function setupLayerControlEvents() {
    const baseLocationsBtn = document.getElementById('layer-base-locations');
    const candidatesBtn = document.getElementById('layer-candidates');
    const metricBtns = document.querySelectorAll('.metric-btn');
    
    // Toggle base locations (blue markers loaded at startup)
    if (baseLocationsBtn) {
        baseLocationsBtn.addEventListener('click', () => {
            const isActive = baseLocationsBtn.dataset.active === 'true';
            baseLocationsBtn.dataset.active = !isActive;
            
            // Get exception IDs (origin and destination if set)
            const exceptions = window.markerExceptions || [];
            
            // Toggle base location markers
            Object.entries(markers).forEach(([id, marker]) => {
                // Skip exception markers (always keep them visible)
                if (exceptions.includes(id)) {
                    return;
                }
                
                if (!isActive) {
                    marker.addTo(map);
                } else {
                    marker.remove();
                }
            });
        });
    }
    
    // Candidates and Metric buttons (mutually exclusive group)
    const visualizationBtns = [candidatesBtn, ...metricBtns].filter(btn => btn);
    
    visualizationBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const wasActive = btn.dataset.active === 'true';
            
            // Deactivate all visualization buttons
            visualizationBtns.forEach(b => b.dataset.active = 'false');
            
            // Clear and remove all metric visualization layers
            layerGroups.probabilityHeatmap.clearLayers();
            layerGroups.priceHeatmap.clearLayers();
            layerGroups.weightHeatmap.clearLayers();
            layerGroups.scores.clearLayers();
            
            map.removeLayer(layerGroups.probabilityHeatmap);
            map.removeLayer(layerGroups.priceHeatmap);
            map.removeLayer(layerGroups.weightHeatmap);
            map.removeLayer(layerGroups.scores);
            
            // Restore all base candidate markers (they might have been hidden by metric layers)
            if (window.candidateMarkersMap && window.candidateMarkersMap.size > 0) {
                window.candidateMarkersMap.forEach((marker, locationId) => {
                    if (!layerGroups.allCandidates.hasLayer(marker)) {
                        layerGroups.allCandidates.addLayer(marker);
                    }
                });
            }
            
            // If wasn't active, activate this one and show its layer
            if (!wasActive) {
                btn.dataset.active = 'true';
                
                // Check if it's the candidates button or a metric button
                if (btn.id === 'layer-candidates') {
                    // Show all candidates layer (already restored above)
                    map.addLayer(layerGroups.allCandidates);
                } else {
                    // It's a metric button, show scaled visualization
                    // First ensure allCandidates layer is on the map (will be hidden selectively)
                    map.addLayer(layerGroups.allCandidates);
                    updateMetricVisualization();
                }
            } else {
                // Was active and now deactivating - remove allCandidates layer
                map.removeLayer(layerGroups.allCandidates);
            }
        });
    });
}

/**
 * Update metric visualization based on active metric button
 */
function updateMetricVisualization() {
    const activeMetricBtn = document.querySelector('.metric-btn[data-active="true"]');
    
    if (!activeMetricBtn || !currentCandidates || currentCandidates.length === 0) {
        return;
    }
    
    const metricType = activeMetricBtn.id.replace('layer-', '');
    const metricMap = {
        'probability': { key: 'probability', layer: 'probabilityHeatmap', color: '#8B5CF6' },
        'price': { key: 'price', layer: 'priceHeatmap', color: '#F59E0B' },
        'weight': { key: 'weight', layer: 'weightHeatmap', color: '#10B981' },
        'score': { key: 'score', layer: 'scores', color: '#3B82F6' }
    };
    
    const config = metricMap[metricType];
    if (!config) return;
    
    // Clear ALL candidate visualization layers first to prevent ghost markers
    layerGroups.allCandidates.clearLayers();
    layerGroups.feasibleCandidates.clearLayers();
    layerGroups.probabilityHeatmap.clearLayers();
    layerGroups.priceHeatmap.clearLayers();
    layerGroups.weightHeatmap.clearLayers();
    layerGroups.scores.clearLayers();
    
    // Draw scaled metric circles
    drawScaledMetricCandidates(currentCandidates, metricType, config.layer, config.color);
    map.addLayer(layerGroups[config.layer]);
}

/**
 * Add layer control for toggling visualization layers
 */
function addLayerControl() {
    const LayerControl = L.Control.extend({
        options: {
            position: 'topright'
        },
        
        onAdd: function(map) {
            const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control layer-control-icons');
            
            container.innerHTML = `
                <button class="layer-icon-btn" id="layer-base-locations" data-active="true" title="Mostrar/ocultar localizaciones base">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                    </svg>
                </button>
                <button class="layer-icon-btn" id="layer-candidates" data-active="false" title="Mostrar/ocultar candidatos evaluados" style="display: none;">
                    <img src="https://files.svgcdn.io/hugeicons/view.svg" alt="View" style="width: 24px; height: 24px;">
                </button>
                <button class="layer-icon-btn metric-btn" id="layer-probability" data-active="false" title="Predicción de probabilidad" style="display: none;">
                    <img src="https://files.svgcdn.io/hugeicons/package.svg" alt="Package" style="width: 24px; height: 24px;">
                </button>
                <button class="layer-icon-btn metric-btn" id="layer-price" data-active="false" title="Predicción de precio" style="display: none;">
                    <img src="https://files.svgcdn.io/hugeicons/money-bag-02.svg" alt="Money" style="width: 24px; height: 24px;">
                </button>
                <button class="layer-icon-btn metric-btn" id="layer-weight" data-active="false" title="Predicción de peso" style="display: none;">
                    <img src="https://files.svgcdn.io/hugeicons/weight-scale-01.svg" alt="Weight" style="width: 24px; height: 24px;">
                </button>
                <button class="layer-icon-btn metric-btn" id="layer-score" data-active="false" title="Score de optimización" style="display: none;">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>
                    </svg>
                </button>
                <button class="layer-icon-btn" id="toggle-results-btn" title="Mostrar/Ocultar resultados" style="display: none;">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" transform="scale(-1, 1)">
                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                        <line x1="9" y1="3" x2="9" y2="21"/>
                        <line x1="14" y1="8" x2="19" y2="8"/>
                        <line x1="14" y1="12" x2="19" y2="12"/>
                        <line x1="14" y1="16" x2="19" y2="16"/>
                    </svg>
                </button>
            `;
            
            // Prevent map interactions when clicking control
            L.DomEvent.disableClickPropagation(container);
            L.DomEvent.disableScrollPropagation(container);
            
            return container;
        }
    });
    
    const layerControl = new LayerControl();
    layerControl.addTo(map);
    
    // Add event listeners
    setTimeout(() => {
        setupLayerControlEvents();
    }, 100);
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
        
        // Validate form after clearing
        validateForm();
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
        
        // Validate form after clearing
        validateForm();
    });
}

/**
 * Initialize form validation
 */
function initFormValidation() {
    // Only need to validate origin and destination
    // Other fields have valid defaults
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    
    // Add event listeners only to origin/destination
    originInput.addEventListener('input', validateForm);
    destinationInput.addEventListener('input', validateForm);
    
    // Initial validation
    validateForm();
}

/**
 * Validate all form fields and enable/disable submit button
 */
function validateForm() {
    const originInput = document.getElementById('origin');
    const destinationInput = document.getElementById('destination');
    const submitBtn = document.getElementById('optimize-btn');
    
    // Only validate that origin and destination are selected
    // All other fields have valid default values
    const isValid = 
        originInput.dataset.locationId && 
        destinationInput.dataset.locationId;
    
    console.log('Validating form:', {
        origin: originInput.dataset.locationId,
        destination: destinationInput.dataset.locationId,
        isValid
    });
    
    submitBtn.disabled = !isValid;
}

/**
 * Show all markers on the map
 */
function showAllMarkers() {
    // Clear marker exceptions
    window.markerExceptions = [];
    
    // Add all markers to map and reset their style
    Object.values(markers).forEach(circle => {
        circle.addTo(map);
        circle.setStyle({ 
            radius: 4,
            fillColor: '#3498db',
            color: '#2980b9',
            weight: 1,
            opacity: 0.8, 
            fillOpacity: 0.6 
        });
    });
    
    // Update the base locations button state to "active"
    const baseLocationsBtn = document.getElementById('layer-base-locations');
    if (baseLocationsBtn) {
        baseLocationsBtn.dataset.active = 'true';
    }
    
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
    // Store the exception IDs for later use
    window.markerExceptions = exceptIds;
    
    // Remove all markers from map, then add back only the exceptions
    Object.entries(markers).forEach(([id, circle]) => {
        circle.remove();
    });
    
    // Add back only the exception markers
    exceptIds.forEach(id => {
        if (markers[id]) {
            markers[id].addTo(map);
        }
    });
    
    // Update the base locations button state to "inactive" (since we're hiding them)
    const baseLocationsBtn = document.getElementById('layer-base-locations');
    if (baseLocationsBtn) {
        baseLocationsBtn.dataset.active = 'false';
    }
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
        originInput.value = formatLocationName(location.name);
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
        
        showStatus('Origen seleccionado. Ahora selecciona el destino.', 'success', 3000);
        
        // Validate form to enable/disable button
        validateForm();
        
        // Check if destination is already set
        updateRouteDisplay();
    }
    // If origin is set but destination is not, set destination
    else if (!destinationInput.dataset.locationId) {
        // Don't allow same location
        if (location.id === originInput.dataset.locationId) {
            showStatus('El destino debe ser diferente al origen', 'warning', 3000);
            return;
        }
        
        destinationInput.value = formatLocationName(location.name);
        destinationInput.dataset.locationId = location.id;
        
        showStatus('Destino seleccionado. Puedes calcular la ruta.', 'success', 3000);
        
        // Validate form to enable/disable button
        validateForm();
        
        // Update route display (hide markers, draw line, zoom)
        updateRouteDisplay();
    }
    // Both are set, do nothing or show message
    else {
        showStatus('Origen y destino ya están seleccionados. Usa el icono de basura para cambiar.', 'info', 3000);
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
                const formattedName = formatLocationName(loc.name);
                const highlightedName = highlightMatch(formattedName, query);
                return `<div class="autocomplete-suggestion" data-index="${index}" data-id="${loc.id}" data-name="${formattedName}">
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
            showStatus('El destino debe ser diferente al origen', 'warning', 3000);
            validateForm();
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
    
    // Validate form after selection
    validateForm();
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
 * Initialize load range slider
 */
function initLoadSlider() {
    const slider = document.getElementById('load');
    const valueDisplay = document.getElementById('load-value');
    
    // Update display when slider changes
    slider.addEventListener('input', (e) => {
        valueDisplay.textContent = parseInt(e.target.value).toLocaleString('es-ES');
        updateSliderBackground(slider);
    });
    
    // Initial background update and display
    valueDisplay.textContent = parseInt(slider.value).toLocaleString('es-ES');
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
        
        showStatus(`${locations.length} localizaciones cargadas`, 'success', 2000);
        
    } catch (error) {
        console.error('Error loading locations:', error);
        showStatus(`ERROR: ${error.message}`, 'error', 0);
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
    showStatus('Optimizando ruta...', 'info');
    const button = event.target.querySelector('button[type="submit"]');
    button.disabled = true;
    button.innerHTML = '<span class="loading"></span> Optimizando...';
    
    try {
        // Prepare data for inference API
        const requestData = {
            origin_id: data.origin,
            destination_id: data.destination,
            truck_type: data.truck_type,
            date: data.date,
            max_detour_km: parseFloat(data.buffer),
            max_candidates: 10,
            truck_capacity_kg: 20000,
            buffer_value_km: parseFloat(data.buffer),
            available_capacity_kg: parseFloat(data.load)
        };
        
        console.log('Calling inference API with:', requestData);
        
        // Call inference API
        const response = await runInference(requestData);
        
        console.log('Inference response:', response);
        
        showStatus(`✅ Ruta optimizada: ${data.origin_name} → ${data.destination_name}`, 'success', 5000);
        
        // Display results on map
        displayInferenceResults(response, data);
        
    } catch (error) {
        console.error('Error optimizing route:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        // Re-validate form to re-enable button if all fields are still valid
        validateForm();
        button.innerHTML = '<img src="/static/img/artificial-intelligence_3489001-white.png" alt="AI" class="ai-icon"> Optimizar Ruta';
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
 * Display inference results on map
 */
function displayInferenceResults(response, formData) {
    const resultsDiv = document.getElementById('results');
    
    // Don't auto-show the panel, just populate the data
    // Format truck type
    const truckType = formData.truck_type === 'refrigerated' ? '❄ Refrigerado' : '🚚 Normal';
    
    // Get routes and candidates from response
    const routes = response.alternative_routes || [];
    const candidates = response.candidates_information || [];
    
    // Store candidates globally for metric visualization
    currentCandidates = candidates;
    
    // Display summary (without h3 since it's in the header now)
    resultsDiv.innerHTML = `
        <div class="result-summary">
            <p><strong>Rutas optimizadas:</strong> ${routes.length}</p>
            <p><strong>Candidatos analizados:</strong> ${candidates.length}</p>
            <p><strong>Viaje base:</strong> ${response.base_trip?.distance?.toFixed(2) || 'N/A'} km</p>
        </div>
        
        <h4>Parámetros de Búsqueda</h4>
        <ul>
            <li><strong>Origen:</strong> ${formData.origin_name}</li>
            <li><strong>Destino:</strong> ${formData.destination_name}</li>
            <li><strong>Tipo de Camión:</strong> ${truckType}</li>
            <li><strong>Desvío máximo:</strong> ${formData.buffer} km</li>
            <li><strong>Carga disponible:</strong> ${formData.load} kg</li>
            <li><strong>Fecha:</strong> ${formData.date}</li>
        </ul>
        
        <h4>Rutas Alternativas (${routes.length})</h4>
        ${routes.length === 0 ? '<p>No se encontraron rutas alternativas.</p>' : routes.map((route, idx) => `
            <div class="route-card">
                <h5>Ruta ${idx + 1}</h5>
                <p><strong>Distancia total:</strong> ${route.total_distance?.toFixed(2) || 'N/A'} km</p>
                <p><strong>Peso total:</strong> ${route.total_weight?.toFixed(2) || 'N/A'} kg</p>
                <p><strong>Precio estimado:</strong> ${route.total_price?.toFixed(2) || 'N/A'} €</p>
                <p><strong>Paradas:</strong> ${route.stops?.length || 0}</p>
                ${route.stops && route.stops.length > 0 ? `
                    <details>
                        <summary>Ver paradas (${route.stops.length})</summary>
                        <ul>
                            ${route.stops.map(stop => `
                                <li>${stop.location_name || stop.location_id} - ${stop.weight?.toFixed(2) || 'N/A'} kg</li>
                            `).join('')}
                        </ul>
                    </details>
                ` : ''}
            </div>
        `).join('')}
    `;
    
    // Draw routes on map using layer visualization
    try {
        // Draw base route with truck type
        if (typeof drawBaseRoute === 'function' && response.base_trip) {
            console.log('Drawing base route on map...');
            drawBaseRoute(response.base_trip, formData.truck_type);
        }
        
        if (typeof drawAllCandidates === 'function' && candidates.length > 0) {
            console.log('Drawing candidates on map...');
            drawAllCandidates(candidates);
        }
        
        if (typeof drawAlternativeRoutes === 'function' && routes.length > 0) {
            console.log('Drawing alternative routes on map...');
            drawAlternativeRoutes(routes, formData.truck_type);
        }
        
        // Show route legend if we have alternative routes
        if (routes.length > 0) {
            showRouteLegend(routes);
        }
    } catch (error) {
        console.error('Error drawing visualization layers:', error);
    }
    
    // Store candidates globally for metric visualization
    currentCandidates = candidates;
    
    // Show the toggle button (don't auto-open the panel)
    const toggleBtn = document.getElementById('toggle-results-btn');
    if (toggleBtn) {
        toggleBtn.style.display = 'flex';
    }
    
    // Show candidates and metric buttons in layer control when results are available
    const candidatesBtn = document.getElementById('layer-candidates');
    if (candidatesBtn) {
        candidatesBtn.style.display = 'flex';
        // Activate candidates button by default and ensure allCandidates layer is visible
        candidatesBtn.dataset.active = 'true';
        map.addLayer(layerGroups.allCandidates);
    }
    
    const metricBtns = document.querySelectorAll('.metric-btn');
    metricBtns.forEach(btn => {
        btn.style.display = 'flex';
        btn.dataset.active = 'false'; // Ensure all metrics start inactive
    });
}

/**
 * Toggle results panel visibility
 */
function toggleResultsPanel() {
    const panel = document.getElementById('results-panel');
    panel.classList.toggle('active');
}

/**
 * Close results panel
 */
function closeResultsPanel() {
    const panel = document.getElementById('results-panel');
    panel.classList.remove('active');
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

/**
 * Show route legend at bottom of map
 * @param {Array} routes - Array of alternative routes
 */
function showRouteLegend(routes) {
    const legend = document.getElementById('route-legend');
    const itemsContainer = document.getElementById('route-legend-items');
    
    if (!legend || !itemsContainer) return;
    
    // Route colors (same as in layers.js) - colorblind-friendly
    const colors = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#ECE133'];
    
    // Clear existing items
    itemsContainer.innerHTML = '';
    
    // Create legend item for each route
    routes.forEach((route, idx) => {
        const color = colors[idx % colors.length];
        
        // Calculate estimated time based on distance (avg 60 km/h)
        const distanceKm = route.total_distance_km || route.total_distance || 0;
        const estimatedMinutes = Math.round((distanceKm / 60) * 60);
        const hours = Math.floor(estimatedMinutes / 60);
        const minutes = estimatedMinutes % 60;
        const timeText = hours > 0 ? `${hours}h ${minutes}min` : `${minutes}min`;
        
        const item = document.createElement('div');
        item.className = 'route-legend-item';
        item.innerHTML = `
            <div class="route-legend-color" style="background-color: ${color};"></div>
            <div class="route-legend-info">
                <div class="route-legend-label">Ruta ${idx + 1}</div>
                <div class="route-legend-stats">${timeText} • ${distanceKm.toFixed(1)} km</div>
            </div>
        `;
        
        itemsContainer.appendChild(item);
    });
    
    // Show legend
    legend.style.display = 'block';
}

/**
 * Hide route legend
 */
function hideRouteLegend() {
    const legend = document.getElementById('route-legend');
    if (legend) {
        legend.style.display = 'none';
    }
}
