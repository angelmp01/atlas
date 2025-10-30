/**
 * ATLAS API Client
 * Handles communication with atlas_api backend
 */

/**
 * Call the inference endpoint
 * @param {Object} params - Inference parameters
 * @returns {Promise<Object>} - Inference response
 */
async function runInference(params) {
    const url = `${API_BASE_URL}/inference`;
    
    console.log('📤 Calling inference API:', url, params);
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('📥 Inference response received:', data);
        
        return data;
        
    } catch (error) {
        console.error('❌ API Error:', error);
        throw error;
    }
}

/**
 * Fetch all locations (for base layer)
 * @returns {Promise<Array>} - List of locations
 */
async function fetchAllLocations() {
    const url = `${API_BASE_URL}/locations`;
    
    try {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log(`📥 Fetched ${data.locations?.length || 0} locations`);
        
        return data.locations || [];
        
    } catch (error) {
        console.warn('⚠️ Could not fetch locations:', error.message);
        return [];
    }
}

console.log('✅ API client loaded');
