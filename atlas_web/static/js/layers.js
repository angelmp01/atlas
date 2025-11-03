/**
 * ATLAS Layers Visualization
 * Functions to render different data layers on the map
 */

/**
 * Format currency in European style (1.234,56 €)
 * @param {number} amount - Amount to format
 * @returns {string} - Formatted currency string
 */
function formatEuroCurrency(amount) {
    if (amount === null || amount === undefined) return '0,00 €';
    return amount.toLocaleString('es-ES', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }) + ' €';
}

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

/**
 * Main function to visualize inference results
 * @param {Object} data - Inference response from API
 * @param {string} truckType - Type of truck ('normal' or 'refrigerated')
 */
function visualizeResults(data, truckType = 'normal') {
    console.log('🎨 Visualizing results...');
    
    // Draw base route (O→D)
    if (data.base_trip) {
        drawBaseRoute(data.base_trip, truckType);
    }
    
    // Draw candidates
    if (data.candidates_information && data.candidates_information.length > 0) {
        drawAllCandidates(data.candidates_information);
        // Note: Metric layers (probability, price, weight, score) are drawn on-demand
        // when user clicks layer control buttons, not automatically
    }
    
    // Draw alternative routes
    if (data.alternative_routes && data.alternative_routes.length > 0) {
        drawAlternativeRoutes(data.alternative_routes);
    }
    
    // Fit map to show all content
    fitMapToBounds(data);
    
    console.log('✅ Visualization complete');
}

/**
 * Calculate scaled radius based on metric value and min/max range
 * @param {number} value - Current metric value
 * @param {number} min - Minimum value in dataset
 * @param {number} max - Maximum value in dataset
 * @returns {number} Scaled radius (10-30px)
 */
function calculateScaledRadius(value, min, max) {
    const baseSize = 12; // 10-15px range
    const maxSize = baseSize * 3; // 3x = 36px
    
    if (min === max) return baseSize;
    
    const normalized = (value - min) / (max - min);
    return baseSize + (normalized * (maxSize - baseSize));
}

/**
 * Get route color and index for a candidate if it's selected in any route (utility function)
 */
function getCandidateRouteInfo(candidateId, routes) {
    const colors = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#ECE133'];
    
    for (let i = 0; i < routes.length; i++) {
        const route = routes[i];
        if (route.waypoints) {
            // Skip first (origin) and last (destination) waypoints
            const intermediateWaypoints = route.waypoints.slice(1, -1);
            for (const wp of intermediateWaypoints) {
                if (wp.location_id === candidateId) {
                    return {
                        routeIndex: i,
                        color: colors[i % colors.length],
                        routeId: route.route_id
                    };
                }
            }
        }
    }
    return null;
}

/**
 * Create unified tooltip content for candidate with all metrics
 * @param {Object} candidate - Candidate data
 * @param {string|null} highlightMetric - Metric to highlight in bold ('probability', 'price', 'weight', 'score', or null for none)
 * @returns {string} HTML tooltip content
 */
function createCandidateTooltip(candidate, highlightMetric = null) {
    const routeInfo = getCandidateRouteInfo(candidate.location_id, window.currentRoutes || []);
    const routeBadge = routeInfo ? `<span style="color: ${routeInfo.color}; font-weight: 600;"> • Ruta ${routeInfo.routeId}</span>` : '';
    
    // Format numbers
    const formattedPrice = Math.round(candidate.p_price_eur).toLocaleString('de-DE');
    const formattedWeight = Math.round(candidate.p_weight_kg).toLocaleString('de-DE');
    
    // Determine bold style for each metric
    const probStyle = highlightMetric === 'probability' ? 'font-weight: bold;' : '';
    const priceStyle = highlightMetric === 'price' ? 'font-weight: bold;' : '';
    const weightStyle = highlightMetric === 'weight' ? 'font-weight: bold;' : '';
    const scoreStyle = highlightMetric === 'score' ? 'font-weight: bold;' : '';
    
    return `
        <div class="tooltip-content" style="min-width: 200px;">
            <strong>${formatLocationName(candidate.location_name)}</strong>${routeBadge}<br>
            <div style="display: flex; gap: 8px; margin-top: 6px; font-size: 11px; align-items: center;">
                <div style="display: flex; align-items: center; gap: 3px; ${probStyle}">
                    <img src="https://files.svgcdn.io/hugeicons/package.svg" alt="Prob" style="width: 12px; height: 12px; opacity: 0.7;">
                    <span>${(candidate.p_probability * 100).toFixed(0)}%</span>
                </div>
                <div style="display: flex; align-items: center; gap: 3px; ${priceStyle}">
                    <img src="https://files.svgcdn.io/hugeicons/money-bag-02.svg" alt="Precio" style="width: 12px; height: 12px; opacity: 0.7;">
                    <span>${formattedPrice} €</span>
                </div>
                <div style="display: flex; align-items: center; gap: 3px; ${weightStyle}">
                    <img src="https://files.svgcdn.io/hugeicons/weight-scale-01.svg" alt="Peso" style="width: 12px; height: 12px; opacity: 0.7;">
                    <span>${formattedWeight} kg</span>
                </div>
                <div style="display: flex; align-items: center; gap: 3px; ${scoreStyle}">
                    <svg viewBox="0 0 24 24" fill="currentColor" style="width: 12px; height: 12px; color: #F59E0B; opacity: 0.7;">
                        <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>
                    </svg>
                    <span>${candidate.score.toFixed(1)}</span>
                </div>
            </div>
        </div>
    `;
}

/**
 * Draw candidates with circles scaled by a specific metric
 * @param {Array} candidates - List of candidate locations
 * @param {string} metric - Metric to use for scaling ('probability', 'price', 'weight', 'score')
 * @param {string} layerGroupName - Name of the layer group to use
 * @param {string} color - Fill color for circles
 */
function drawScaledMetricCandidates(candidates, metric, layerGroupName, color) {
    const layerGroup = layerGroups[layerGroupName];
    if (!layerGroup) return;
    
    layerGroup.clearLayers();
    
    // Get metric values and calculate min/max
    const metricMap = {
        'probability': 'p_probability',
        'price': 'p_price_eur',
        'weight': 'p_weight_kg',
        'score': 'score'
    };
    
    const metricKey = metricMap[metric];
    if (!metricKey) return;
    
    const values = candidates.map(c => c[metricKey]).filter(v => v != null);
    if (values.length === 0) return;
    
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    
    // Track which location_ids are drawn in this metric layer
    const drawnLocationIds = new Set();
    
    // Draw each candidate with scaled radius
    candidates.forEach(candidate => {
        const value = candidate[metricKey];
        if (value == null) return;
        
        const radius = calculateScaledRadius(value, minValue, maxValue);
        
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: radius,
            fillColor: color,
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.7
        });
        
        // Create unified tooltip with highlighted metric
        const tooltipContent = createCandidateTooltip(candidate, metric);
        
        marker.bindTooltip(tooltipContent, {
            direction: 'top',
            offset: [0, -radius]
        });
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroup);
        drawnLocationIds.add(candidate.location_id);
        
        // Hide the base candidate marker if it exists
        if (window.candidateMarkersMap && window.candidateMarkersMap.has(candidate.location_id)) {
            const baseMarker = window.candidateMarkersMap.get(candidate.location_id);
            layerGroups.allCandidates.removeLayer(baseMarker);
        }
    });
    
    console.log(`✅ Drew ${candidates.length} scaled ${metric} markers (${minValue.toFixed(2)} - ${maxValue.toFixed(2)})`);
    console.log(`🔒 Hidden ${drawnLocationIds.size} base candidate markers to avoid duplicates`);
}

/**
 * Draw base route (direct O→D without deviations)
 * @param {Object} baseTrip - Base trip data
 * @param {string} truckType - Type of truck ('normal' or 'refrigerated')
 */
function drawBaseRoute(baseTrip, truckType = 'normal') {
    layerGroups.baseRoute.clearLayers();
    
    // Validate required fields
    if (!baseTrip.route_geometry) {
        console.warn('No route geometry for base trip');
        return;
    }
    
    // Extract coordinates from origin and destination objects if needed
    const originLat = baseTrip.origin_lat || baseTrip.origin?.latitude;
    const originLon = baseTrip.origin_lon || baseTrip.origin?.longitude;
    const destLat = baseTrip.destination_lat || baseTrip.destination?.latitude;
    const destLon = baseTrip.destination_lon || baseTrip.destination?.longitude;
    const originName = baseTrip.origin_name || baseTrip.origin?.name || baseTrip.origin_id;
    const destName = baseTrip.destination_name || baseTrip.destination?.name || baseTrip.destination_id;
    
    if (!originLat || !originLon || !destLat || !destLon) {
        console.error('Missing origin/destination coordinates in base trip:', baseTrip);
        return;
    }
    
    // Parse GeoJSON LineString
    let coords;
    try {
        const geojson = typeof baseTrip.route_geometry === 'string' 
            ? JSON.parse(baseTrip.route_geometry) 
            : baseTrip.route_geometry;
        
        if (geojson.type === 'LineString' && geojson.coordinates) {
            // GeoJSON uses [lon, lat], Leaflet uses [lat, lon]
            coords = geojson.coordinates.map(c => [c[1], c[0]]);
        }
    } catch (e) {
        console.error('Error parsing base route geometry:', e);
        return;
    }
    
    if (!coords || coords.length === 0) {
        console.warn('No valid coordinates in base route geometry');
        return;
    }
    
    // Draw route line
    const routeLine = L.polyline(coords, {
        color: '#3B82F6',
        weight: 4,
        opacity: 0.7,
        dashArray: '10, 10',
        lineJoin: 'round'
    }).addTo(layerGroups.baseRoute);
    
    routeLine.bindPopup(`
        <div class="popup-content">
            <h3>🚛 Ruta Base</h3>
            <p><strong>Origen:</strong> ${formatLocationName(originName)}</p>
            <p><strong>Destino:</strong> ${formatLocationName(destName)}</p>
            <p><strong>Distancia:</strong> ${baseTrip.distance_km.toFixed(1)} km</p>
        </div>
    `);
    
    // Origin marker - use emoji truck icon
    L.marker([originLat, originLon], {
        icon: createCustomIcon('🚛', '#3B82F6', 'large')
    }).bindPopup(`
        <div class="popup-content">
            <h3>🚛 Origen</h3>
            <p><strong>${formatLocationName(originName)}</strong></p>
        </div>
    `).addTo(layerGroups.baseRoute);
    
    // Destination marker - flag emoji icon
    L.marker([destLat, destLon], {
        icon: createCustomIcon('🏁', '#10B981', 'large')
    }).bindPopup(`
        <div class="popup-content">
            <h3>🏁 Destino</h3>
            <p><strong>${formatLocationName(destName)}</strong></p>
        </div>
    `).addTo(layerGroups.baseRoute);
    
    console.log('✅ Base route drawn');
}

// Global map to store candidate markers by location_id for selective hiding
window.candidateMarkersMap = new Map();

/**
 * Draw all candidates as simple markers
 */
function drawAllCandidates(candidates) {
    layerGroups.allCandidates.clearLayers();
    window.candidateMarkersMap.clear();
    
    candidates.forEach(candidate => {
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: 6,
            fillColor: '#F59E0B',
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.7
        });
        
        // Use unified tooltip (no metric highlighted)
        const tooltipContent = createCandidateTooltip(candidate, null);
        
        marker.bindTooltip(tooltipContent, {
            direction: 'top',
            offset: [0, -10]
        });
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.allCandidates);
        
        // Store marker reference by location_id
        window.candidateMarkersMap.set(candidate.location_id, marker);
    });
    
    console.log(`✅ Drew ${candidates.length} candidate markers`);
}

/**
 * Draw feasible vs non-feasible candidates (different colors)
 */
function drawFeasibleCandidates(candidates) {
    layerGroups.feasibleCandidates.clearLayers();
    
    candidates.forEach(candidate => {
        const isFeasible = candidate.is_feasible;
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: 8,
            fillColor: isFeasible ? '#10B981' : '#EF4444',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                ${isFeasible ? '✅ Factible' : '❌ No factible'}<br>
                Δd: ${candidate.delta_d_km.toFixed(1)} km
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.feasibleCandidates);
    });
    
    console.log('✅ Drew feasible/non-feasible markers');
}

/**
 * Draw probability heatmap (circle size = probability)
 */
function drawProbabilityHeatmap(candidates) {
    layerGroups.probabilityHeatmap.clearLayers();
    
    // Find min/max for normalization
    const probs = candidates.map(c => c.p_probability);
    const minProb = Math.min(...probs);
    const maxProb = Math.max(...probs);
    
    candidates.forEach(candidate => {
        const normalized = (candidate.p_probability - minProb) / (maxProb - minProb || 1);
        const radius = 5 + normalized * 20; // 5-25px
        const opacity = 0.3 + normalized * 0.6; // 0.3-0.9
        
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: radius,
            fillColor: getColorGradient(normalized, 'green'),
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: opacity
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                Probabilidad: ${(candidate.p_probability * 100).toFixed(1)}%
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.probabilityHeatmap);
    });
    
    console.log('✅ Drew probability heatmap');
}

/**
 * Draw price heatmap (circle size = price)
 */
function drawPriceHeatmap(candidates) {
    layerGroups.priceHeatmap.clearLayers();
    
    const prices = candidates.map(c => c.p_price_eur);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    candidates.forEach(candidate => {
        const normalized = (candidate.p_price_eur - minPrice) / (maxPrice - minPrice || 1);
        const radius = 5 + normalized * 20;
        const opacity = 0.3 + normalized * 0.6;
        
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: radius,
            fillColor: getColorGradient(normalized, 'yellow'),
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: opacity
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                Precio: ${candidate.p_price_eur.toFixed(0)}€
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.priceHeatmap);
    });
    
    console.log('✅ Drew price heatmap');
}

/**
 * Draw weight heatmap (circle size = weight)
 */
function drawWeightHeatmap(candidates) {
    layerGroups.weightHeatmap.clearLayers();
    
    const weights = candidates.map(c => c.p_weight_kg);
    const minWeight = Math.min(...weights);
    const maxWeight = Math.max(...weights);
    
    candidates.forEach(candidate => {
        const normalized = (candidate.p_weight_kg - minWeight) / (maxWeight - minWeight || 1);
        const radius = 5 + normalized * 20;
        const opacity = 0.3 + normalized * 0.6;
        
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: radius,
            fillColor: getColorGradient(normalized, 'purple'),
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: opacity
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                Peso: ${candidate.p_weight_kg.toFixed(0)} kg
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.weightHeatmap);
    });
    
    console.log('✅ Drew weight heatmap');
}

/**
 * Draw score visualization (color gradient red→green)
 */
function drawScores(candidates) {
    layerGroups.scores.clearLayers();
    
    const scores = candidates.map(c => c.score);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    
    // Sort to identify top candidates
    const sorted = [...candidates].sort((a, b) => b.score - a.score);
    
    candidates.forEach((candidate, index) => {
        const normalized = (candidate.score - minScore) / (maxScore - minScore || 1);
        const radius = 5 + normalized * 20;
        const isTop10 = sorted.indexOf(candidate) < 10;
        
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: radius,
            fillColor: getColorGradient(normalized, 'score'),
            color: isTop10 ? '#000' : '#fff',
            weight: isTop10 ? 3 : 1,
            opacity: 1,
            fillOpacity: 0.7
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                Score: ${candidate.score.toFixed(2)}<br>
                Score/km: ${candidate.score_per_km.toFixed(2)}<br>
                ${isTop10 ? '<strong>🏆 Top 10</strong>' : ''}
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.scores);
    });
    
    console.log('✅ Drew score visualization');
}

/**
 * Draw ETA heatmap (color = distance from origin)
 */
function drawETAHeatmap(candidates) {
    layerGroups.etaHeatmap.clearLayers();
    
    const etas = candidates.map(c => c.eta_km);
    const minETA = Math.min(...etas);
    const maxETA = Math.max(...etas);
    
    candidates.forEach(candidate => {
        const normalized = (candidate.eta_km - minETA) / (maxETA - minETA || 1);
        const radius = 5 + (1 - normalized) * 15; // Closer = larger
        
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: radius,
            fillColor: getColorGradient(1 - normalized, 'eta'), // Green=close, red=far
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.7
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                ETA: ${candidate.eta_km.toFixed(1)} km<br>
                f_eta: ${candidate.f_eta.toFixed(4)}
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.etaHeatmap);
    });
    
    console.log('✅ Drew ETA heatmap');
}

/**
 * Draw delta distance indicators (arrows showing if route shortens/lengthens)
 */
function drawDeltaDistance(candidates) {
    layerGroups.deltaDistance.clearLayers();
    
    candidates.forEach(candidate => {
        const delta = candidate.delta_d_km;
        const isShortcut = delta < 0;
        
        // Draw arrow from candidate towards destination
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: Math.min(Math.abs(delta) / 2 + 5, 20),
            fillColor: isShortcut ? '#10B981' : '#EF4444',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.6
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${formatLocationName(candidate.location_name)}</strong><br>
                Δ distancia: ${delta.toFixed(1)} km<br>
                ${isShortcut ? '✅ Acorta ruta' : '⚠️ Alarga ruta'}
            </div>
        `);
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.deltaDistance);
    });
    
    console.log('✅ Drew delta distance indicators');
}

/**
 * Draw alternative routes (colored polylines)
 * @param {Array} routes - Array of alternative route objects
 * @param {string} truckType - Type of truck ('normal' or 'refrigerated')
 */
function drawAlternativeRoutes(routes, truckType = 'normal', candidates = []) {
    layerGroups.alternativeRoutes.clearLayers();
    
    // Colorblind-friendly palette (blue, orange, magenta)
    const colors = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#ECE133'];
    
    routes.forEach((route, index) => {
        const color = colors[index % colors.length];
        
        // Draw route using road geometry if available, otherwise straight lines
        let routeLine;
        if (route.route_geometry) {
            // Parse GeoJSON and create polyline from actual road network
            try {
                const geojson = JSON.parse(route.route_geometry);
                if (geojson.type === 'LineString' && geojson.coordinates.length > 0) {
                    // GeoJSON uses [lon, lat], Leaflet uses [lat, lon]
                    const coords = geojson.coordinates.map(c => [c[1], c[0]]);
                    routeLine = L.polyline(coords, {
                        color: color,
                        weight: 5,
                        opacity: 0.8,
                        lineJoin: 'round'
                    });
                }
            } catch (e) {
                console.warn('Error parsing route geometry, falling back to straight lines:', e);
            }
        }
        
        // Fallback: draw straight lines between waypoints
        if (!routeLine) {
            const coords = route.waypoints.map(w => [w.latitude, w.longitude]);
            routeLine = L.polyline(coords, {
                color: color,
                weight: 5,
                opacity: 0.8,
                lineJoin: 'round'
            });
        }
        
        routeLine.bindPopup(`
            <div class="popup-content">
                <h3 style="color: ${color}">🛣️ Ruta ${index + 1}</h3>
                <p><strong>Score total:</strong> ${route.total_score.toFixed(2)}</p>
                <p><strong>Distancia:</strong> ${route.total_distance_km.toFixed(1)} km</p>
                <p><strong>Desvío extra:</strong> +${route.extra_distance_km.toFixed(1)} km</p>
                <p><strong>Peso esperado:</strong> ${route.total_expected_weight_kg.toFixed(0)} kg</p>
                <p><strong>💰 Ingresos esperados:</strong> ${formatEuroCurrency(route.total_expected_revenue_eur)}</p>
                <hr>
                <p><strong>Paradas:</strong></p>
                <ul>
                    ${route.waypoints.map(w => `<li>${formatLocationName(w.location_name)}</li>`).join('')}
                </ul>
            </div>
        `);
        
        routeLine.on('click', () => showRouteDetails(route, index));
        
        routeLine.addTo(layerGroups.alternativeRoutes);
        
        // Draw numbered waypoint markers
        route.waypoints.forEach((waypoint, wpIndex) => {
            const isOrigin = wpIndex === 0;
            const isDestination = wpIndex === route.waypoints.length - 1;
            
            let icon;
            if (isOrigin) {
                icon = createCustomIcon('🚛', color, 'medium');
            } else if (isDestination) {
                icon = createCustomIcon('🏁', color, 'medium');
            } else {
                icon = createCustomIcon(wpIndex.toString(), color, 'small');
            }
            
            const marker = L.marker([waypoint.latitude, waypoint.longitude], { icon });
            
            // For intermediate waypoints (not origin/destination), find candidate data and show unified tooltip
            if (!isOrigin && !isDestination && candidates.length > 0) {
                const candidate = candidates.find(c => c.location_id === waypoint.location_id);
                
                if (candidate) {
                    const tooltipContent = createCandidateTooltip(candidate, null);
                    
                    // Bind tooltip (hover)
                    marker.bindTooltip(tooltipContent, {
                        permanent: false,
                        direction: 'top',
                        offset: [0, -10]
                    });
                    
                    // Bind popup (click - for mobile)
                    marker.bindPopup(tooltipContent);
                } else {
                    // Fallback if candidate not found
                    const fallbackContent = `
                        <div style="font-weight: bold;">Parada ${wpIndex}</div>
                        <div>${waypoint.location_name}</div>
                    `;
                    marker.bindTooltip(fallbackContent, {
                        permanent: false,
                        direction: 'top',
                        offset: [0, -10]
                    });
                    marker.bindPopup(fallbackContent);
                }
            } else {
                // Origin and destination keep simple display
                const simpleContent = `
                    <div style="font-weight: bold; color: ${color};">
                        ${isOrigin ? '🚛 Origen' : '🏁 Destino'}
                    </div>
                    <div>${waypoint.location_name}</div>
                `;
                marker.bindTooltip(simpleContent, {
                    permanent: false,
                    direction: 'top',
                    offset: [0, -10]
                });
                marker.bindPopup(simpleContent);
            }
            
            marker.addTo(layerGroups.alternativeRoutes);
        });
    });
    
    console.log(`✅ Drew ${routes.length} alternative routes`);
}

/**
 * Create custom icon with emoji/text
 */
function createCustomIcon(text, color, size) {
    const sizes = {
        small: 24,
        medium: 32,
        large: 40
    };
    
    const iconSize = sizes[size] || sizes.medium;
    
    return L.divIcon({
        html: `<div style="
            background-color: ${color};
            color: white;
            border: 2px solid white;
            border-radius: 50%;
            width: ${iconSize}px;
            height: ${iconSize}px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: ${iconSize * 0.5}px;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        ">${text}</div>`,
        className: '',
        iconSize: [iconSize, iconSize],
        iconAnchor: [iconSize / 2, iconSize / 2]
    });
}

/**
 * Get color for normalized value (0-1) based on gradient type
 */
function getColorGradient(value, type) {
    // Clamp value between 0 and 1
    value = Math.max(0, Math.min(1, value));
    
    switch(type) {
        case 'green':
            // Light green to dark green
            const r_g = Math.round(220 - value * 140);
            const g_g = Math.round(255 - value * 90);
            const b_g = Math.round(220 - value * 140);
            return `rgb(${r_g}, ${g_g}, ${b_g})`;
        
        case 'yellow':
            // Light yellow to orange
            return value < 0.5 
                ? `rgb(255, ${Math.round(255 - value * 100)}, 100)`
                : `rgb(255, ${Math.round(205 - (value - 0.5) * 150)}, ${Math.round(100 - (value - 0.5) * 100)})`;
        
        case 'purple':
            // Light purple to dark purple
            const r_p = Math.round(200 - value * 60);
            const b_p = Math.round(255 - value * 100);
            return `rgb(${r_p}, 100, ${b_p})`;
        
        case 'score':
            // Red → Yellow → Green
            if (value < 0.5) {
                // Red to yellow
                return `rgb(255, ${Math.round(value * 510)}, 0)`;
            } else {
                // Yellow to green
                return `rgb(${Math.round(255 - (value - 0.5) * 510)}, 255, 0)`;
            }
        
        case 'eta':
            // Green (close) to Red (far)
            return value < 0.5
                ? `rgb(${Math.round(value * 510)}, 255, 0)`
                : `rgb(255, ${Math.round(255 - (value - 0.5) * 510)}, 0)`;
        
        default:
            return '#3B82F6';
    }
}

/**
 * Fit map bounds to show all data
 */
function fitMapToBounds(data) {
    const bounds = L.latLngBounds();
    
    // Add base trip points
    if (data.base_trip) {
        bounds.extend([data.base_trip.origin_lat, data.base_trip.origin_lon]);
        bounds.extend([data.base_trip.destination_lat, data.base_trip.destination_lon]);
    }
    
    // Add all candidates
    if (data.candidates_information) {
        data.candidates_information.forEach(c => {
            bounds.extend([c.latitude, c.longitude]);
        });
    }
    
    if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

/**
 * Show all candidates in results panel, ordered by distance to origin
 */
function showAllCandidatesPanel(candidates, routes, origin) {
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results');
    const panelTitle = document.getElementById('results-panel-title');
    
    if (!candidates || candidates.length === 0) {
        return;
    }
    
    // Update panel title
    panelTitle.textContent = `Candidatos Evaluados (${candidates.length})`;
    
    // Sort candidates by ETA (distance to origin)
    const sortedCandidates = [...candidates].sort((a, b) => a.eta_km - b.eta_km);
    
    const colors = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#ECE133'];
    
    resultsContent.innerHTML = `
        <div class="candidates-list">
            ${sortedCandidates.map(candidate => {
                // Check if candidate is selected in any route
                const routeInfo = getCandidateRouteInfo(candidate.location_id, routes || []);
                const backgroundColor = routeInfo ? `${routeInfo.color}15` : 'transparent';
                const borderColor = routeInfo ? routeInfo.color : '#e0e0e0';
                
                // Format numbers with thousands separator (force dot separator)
                const formattedPrice = Math.round(candidate.p_price_eur).toLocaleString('de-DE');
                const formattedWeight = Math.round(candidate.p_weight_kg).toLocaleString('de-DE');
                
                return `
                    <div class="candidate-item" style="background-color: ${backgroundColor}; border-left: 3px solid ${borderColor};" 
                         data-candidate-id="${candidate.location_id}">
                        <div class="candidate-name">
                            <strong>${formatLocationName(candidate.location_name)}</strong>
                            ${routeInfo ? `<span style="color: ${routeInfo.color}; font-weight: 600; margin-left: 4px;">Ruta ${routeInfo.routeId}</span>` : ''}
                        </div>
                        <div class="candidate-metrics">
                            <div class="candidate-metric">
                                <img src="https://files.svgcdn.io/hugeicons/package.svg" alt="Probabilidad" style="width: 14px; height: 14px;">
                                <span>${(candidate.p_probability * 100).toFixed(0)}%</span>
                            </div>
                            <div class="candidate-metric">
                                <img src="https://files.svgcdn.io/hugeicons/money-bag-02.svg" alt="Precio" style="width: 14px; height: 14px;">
                                <span>${formattedPrice} €</span>
                            </div>
                            <div class="candidate-metric">
                                <img src="https://files.svgcdn.io/hugeicons/weight-scale-01.svg" alt="Peso" style="width: 14px; height: 14px;">
                                <span>${formattedWeight} kg</span>
                            </div>
                            <div class="candidate-metric">
                                <svg viewBox="0 0 24 24" fill="currentColor" style="width: 14px; height: 14px; color: #F59E0B;">
                                    <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>
                                </svg>
                                <span>${candidate.score.toFixed(1)}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
    
    resultsPanel.classList.add('open');
}

/**
 * Show candidate details in results panel
 */
function showCandidateDetails(candidate) {
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results');
    
    // Get route info if candidate is selected
    const routeInfo = getCandidateRouteInfo(candidate.location_id, window.currentRoutes || []);
    const routeIndicator = routeInfo 
        ? `<span style="display: inline-block; width: 12px; height: 12px; background-color: ${routeInfo.color}; border-radius: 50%; margin-left: 8px;" title="Seleccionado en Ruta ${routeInfo.routeId}"></span>`
        : '';
    
    resultsContent.innerHTML = `
        <div class="candidate-details">
            <h4>📍 ${formatLocationName(candidate.location_name)}${routeIndicator}</h4>
            
            <div class="details-section">
                <h5>Predicciones</h5>
                <p>
                    <img src="https://files.svgcdn.io/hugeicons/package.svg" alt="Probabilidad" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 4px;">
                    <strong>Probabilidad:</strong> ${(candidate.p_probability * 100).toFixed(1)}%
                </p>
                <p>
                    <img src="https://files.svgcdn.io/hugeicons/money-bag-02.svg" alt="Precio" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 4px;">
                    <strong>Precio:</strong> ${formatEuroCurrency(candidate.p_price_eur)}
                </p>
                <p>
                    <img src="https://files.svgcdn.io/hugeicons/weight-scale-01.svg" alt="Peso" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 4px;">
                    <strong>Peso:</strong> ${candidate.p_weight_kg.toFixed(0)} kg
                </p>
            </div>
            
            <div class="details-section">
                <h5>Score</h5>
                <p>
                    <svg viewBox="0 0 24 24" fill="currentColor" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 4px; color: #F59E0B;">
                        <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>
                    </svg>
                    <strong>Score final:</strong> ${candidate.score.toFixed(2)}
                </p>
            </div>
        </div>
    `;
    
    resultsPanel.classList.add('open');
}

/**
 * Show route details in results panel
 */
function showRouteDetails(route, index) {
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results');
    
    resultsContent.innerHTML = `
        <div class="route-details">
            <h4>🛣️ Ruta ${index + 1}</h4>
            
            <div class="details-section">
                <h5>Métricas</h5>
                <p><strong>Score total:</strong> ${route.total_score.toFixed(2)}</p>
                <p><strong>Distancia total:</strong> ${route.total_distance_km.toFixed(1)} km</p>
                <p><strong>Desvío extra:</strong> +${route.extra_distance_km.toFixed(1)} km</p>
                <p><strong>Peso esperado:</strong> ${route.total_expected_weight_kg.toFixed(0)} kg</p>
                <p><strong>💰 Ingresos esperados:</strong> ${formatEuroCurrency(route.total_expected_revenue_eur)}</p>
            </div>
            
            <div class="details-section">
                <h5>Paradas (${route.waypoints.length})</h5>
                <ol>
                    ${route.waypoints.map(w => `
                        <li><strong>${formatLocationName(w.location_name)}</strong></li>
                    `).join('')}
                </ol>
            </div>
        </div>
    `;
    
    resultsPanel.classList.add('open');
}

console.log('✅ Layers visualization loaded');

