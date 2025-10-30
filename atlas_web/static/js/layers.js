/**
 * ATLAS Layers Visualization
 * Functions to render different data layers on the map
 */

/**
 * Main function to visualize inference results
 * @param {Object} data - Inference response from API
 */
function visualizeResults(data) {
    console.log('🎨 Visualizing results...');
    
    // Draw base route (O→D)
    if (data.base_trip) {
        drawBaseRoute(data.base_trip);
    }
    
    // Draw candidates
    if (data.candidates_information && data.candidates_information.length > 0) {
        drawAllCandidates(data.candidates_information);
        drawFeasibleCandidates(data.candidates_information);
        drawProbabilityHeatmap(data.candidates_information);
        drawPriceHeatmap(data.candidates_information);
        drawWeightHeatmap(data.candidates_information);
        drawScores(data.candidates_information);
        drawETAHeatmap(data.candidates_information);
        drawDeltaDistance(data.candidates_information);
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
        
        // Format tooltip based on metric
        let metricLabel = metric;
        let metricValue = value.toFixed(2);
        
        if (metric === 'probability') {
            metricLabel = 'Probabilidad';
            metricValue = (value * 100).toFixed(1) + '%';
        } else if (metric === 'price') {
            metricLabel = 'Precio';
            metricValue = value.toFixed(2) + ' €';
        } else if (metric === 'weight') {
            metricLabel = 'Peso';
            metricValue = value.toFixed(2) + ' kg';
        } else if (metric === 'score') {
            metricLabel = 'Score';
        }
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${candidate.location_name}</strong><br>
                ${metricLabel}: ${metricValue}
            </div>
        `, {
            direction: 'top',
            offset: [0, -radius]
        });
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroup);
    });
    
    console.log(`✅ Drew ${candidates.length} scaled ${metric} markers (${minValue.toFixed(2)} - ${maxValue.toFixed(2)})`);
}

/**
 * Draw base route (direct O→D without deviations)
 */
function drawBaseRoute(baseTrip) {
    layerGroups.baseRoute.clearLayers();
    
    if (!baseTrip.route_geometry) {
        console.warn('No route geometry for base trip');
        return;
    }
    
    // Parse LineString coordinates
    const coords = baseTrip.route_geometry.coordinates.map(c => [c[1], c[0]]);
    
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
            <p><strong>Origen:</strong> ${baseTrip.origin_name || baseTrip.origin_id}</p>
            <p><strong>Destino:</strong> ${baseTrip.destination_name || baseTrip.destination_id}</p>
            <p><strong>Distancia:</strong> ${baseTrip.distance_km.toFixed(1)} km</p>
        </div>
    `);
    
    // Origin marker
    L.marker([baseTrip.origin_lat, baseTrip.origin_lon], {
        icon: createCustomIcon('🚛', '#3B82F6', 'large')
    }).bindPopup(`
        <div class="popup-content">
            <h3>🚛 Origen</h3>
            <p><strong>${baseTrip.origin_name || baseTrip.origin_id}</strong></p>
        </div>
    `).addTo(layerGroups.baseRoute);
    
    // Destination marker
    L.marker([baseTrip.destination_lat, baseTrip.destination_lon], {
        icon: createCustomIcon('🏁', '#10B981', 'large')
    }).bindPopup(`
        <div class="popup-content">
            <h3>🏁 Destino</h3>
            <p><strong>${baseTrip.destination_name || baseTrip.destination_id}</strong></p>
        </div>
    `).addTo(layerGroups.baseRoute);
    
    console.log('✅ Base route drawn');
}

/**
 * Draw all candidates as simple markers
 */
function drawAllCandidates(candidates) {
    layerGroups.allCandidates.clearLayers();
    
    candidates.forEach(candidate => {
        const marker = L.circleMarker([candidate.latitude, candidate.longitude], {
            radius: 6,
            fillColor: '#F59E0B',
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.7
        });
        
        marker.bindTooltip(`
            <div class="tooltip-content">
                <strong>${candidate.location_name}</strong><br>
                ETA: ${candidate.eta_km.toFixed(1)} km<br>
                Δd: ${candidate.delta_d_km.toFixed(1)} km
            </div>
        `, {
            direction: 'top',
            offset: [0, -10]
        });
        
        marker.on('click', () => showCandidateDetails(candidate));
        
        marker.addTo(layerGroups.allCandidates);
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
                <strong>${candidate.location_name}</strong><br>
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
                <strong>${candidate.location_name}</strong><br>
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
                <strong>${candidate.location_name}</strong><br>
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
                <strong>${candidate.location_name}</strong><br>
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
                <strong>${candidate.location_name}</strong><br>
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
                <strong>${candidate.location_name}</strong><br>
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
                <strong>${candidate.location_name}</strong><br>
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
 */
function drawAlternativeRoutes(routes) {
    layerGroups.alternativeRoutes.clearLayers();
    
    const colors = ['#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];
    
    routes.forEach((route, index) => {
        const color = colors[index % colors.length];
        
        // Draw route waypoints as line
        const coords = route.waypoints.map(w => [w.latitude, w.longitude]);
        
        const routeLine = L.polyline(coords, {
            color: color,
            weight: 5,
            opacity: 0.8,
            lineJoin: 'round'
        });
        
        routeLine.bindPopup(`
            <div class="popup-content">
                <h3 style="color: ${color}">🛣️ Ruta ${index + 1}</h3>
                <p><strong>Score total:</strong> ${route.total_score.toFixed(2)}</p>
                <p><strong>Distancia:</strong> ${route.total_distance_km.toFixed(1)} km</p>
                <p><strong>Desvío extra:</strong> +${route.extra_distance_km.toFixed(1)} km</p>
                <p><strong>Peso esperado:</strong> ${route.total_expected_weight_kg.toFixed(0)} kg</p>
                <hr>
                <p><strong>Waypoints:</strong></p>
                <ul>
                    ${route.waypoints.map(w => `<li>${w.location_name}</li>`).join('')}
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
            
            marker.bindPopup(`
                <div class="popup-content">
                    <h3 style="color: ${color}">Waypoint ${wpIndex + 1}</h3>
                    <p><strong>${waypoint.location_name}</strong></p>
                </div>
            `);
            
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
 * Show candidate details in side panel
 */
function showCandidateDetails(candidate) {
    const sidePanel = document.getElementById('side-panel');
    const content = document.getElementById('side-panel-content');
    const title = document.getElementById('side-panel-title');
    
    title.textContent = `📍 ${candidate.location_name}`;
    
    content.innerHTML = `
        <div class="details-section">
            <h4>Ubicación</h4>
            <p><strong>ID:</strong> ${candidate.location_id}</p>
            <p><strong>Coordenadas:</strong> ${candidate.latitude.toFixed(4)}, ${candidate.longitude.toFixed(4)}</p>
        </div>
        
        <div class="details-section">
            <h4>Distancias</h4>
            <p><strong>ETA (O→i):</strong> ${candidate.eta_km.toFixed(1)} km</p>
            <p><strong>Delta (desvío):</strong> ${candidate.delta_d_km.toFixed(1)} km</p>
            <p><strong>f_eta:</strong> ${candidate.f_eta.toFixed(4)}</p>
        </div>
        
        <div class="details-section">
            <h4>Predicciones ML</h4>
            <p><strong>Probabilidad:</strong> ${(candidate.p_probability * 100).toFixed(1)}%</p>
            <p><strong>Precio:</strong> ${candidate.p_price_eur.toFixed(2)}€</p>
            <p><strong>Peso:</strong> ${candidate.p_weight_kg.toFixed(0)} kg</p>
        </div>
        
        <div class="details-section">
            <h4>Score</h4>
            <p><strong>Total:</strong> ${candidate.score.toFixed(2)}</p>
            <p><strong>Por km:</strong> ${candidate.score_per_km.toFixed(2)}</p>
            <p><strong>Factible:</strong> ${candidate.is_feasible ? '✅ Sí' : '❌ No'}</p>
        </div>
    `;
    
    sidePanel.classList.add('open');
}

/**
 * Show route details in side panel
 */
function showRouteDetails(route, index) {
    const sidePanel = document.getElementById('side-panel');
    const content = document.getElementById('side-panel-content');
    const title = document.getElementById('side-panel-title');
    
    title.textContent = `🛣️ Ruta ${index + 1}`;
    
    content.innerHTML = `
        <div class="details-section">
            <h4>Métricas</h4>
            <p><strong>Score total:</strong> ${route.total_score.toFixed(2)}</p>
            <p><strong>Distancia total:</strong> ${route.total_distance_km.toFixed(1)} km</p>
            <p><strong>Desvío extra:</strong> +${route.extra_distance_km.toFixed(1)} km</p>
            <p><strong>Peso esperado:</strong> ${route.total_expected_weight_kg.toFixed(0)} kg</p>
        </div>
        
        <div class="details-section">
            <h4>Waypoints (${route.waypoints.length})</h4>
            <ol>
                ${route.waypoints.map(w => `
                    <li><strong>${w.location_name}</strong></li>
                `).join('')}
            </ol>
        </div>
    `;
    
    sidePanel.classList.add('open');
}

console.log('✅ Layers visualization loaded');
