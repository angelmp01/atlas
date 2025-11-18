// Multilingual translation data
const translations = {
    ca: {
        intro_title: "Sobre ATLAS",
        intro_text: "ATLAS és un projecte desenvolupat al Postgrau en Intel·ligència Artificial aplicada al Transport i la Logística de la UPC. El seu objectiu és ajudar un operador a decidir quins desviaments val la pena incorporar entre un origen i una destinació, maximitzant la rendibilitat del trajecte sense comprometre capacitat ni desviar excessivament la ruta.",
        hybrid_text: "El sistema analitza centenars de possibles parades properes al corredor O→D. Per a avaluar-les construïm un motor híbrid basat en tres pilars.",
        pillar1_text: "El primer és un mòdul de models de Machine Learning que prediu viatges diaris, benefici i pes esperat per a cada relació OD.",
        pillar2_text: "El segon és un motor de scoring que normalitza i combina aquestes prediccions en una mètrica única que afavoreix el rendible i penalitza consums de capacitat o desviaments alts.",
        pillar3_text: "El tercer és un algoritme tipus motxilla (greedy knapsack) que selecciona la combinació òptima de parades respectant les restriccions del camió i el buffer.",
        optimization_text: "Una vegada escollides les millors parades, ATLAS reordena l'itinerari i l'enruta sobre la xarxa de carreteres utilitzant PostGIS + pgRouting, generant diverses alternatives reals O→D amb geometria, temps, distància i detall econòmic.",
        transparency_text: "Un aspecte clau del projecte és que tot el flux és autoexplicatiu i visual: cada etapa pot inspeccionarse sobre un mapa o un panell, permetent entendre quins candidats es van avaluar, quin score obté cada punt i com es construeixen les rutes finals. Això converteix ATLAS en una eina transparent, interpretable i alineada amb escenaris logístics reals.",
        
        footer_text: "L'equip d'ATLAS està format per Gabriel Bermúdez, Íngrid Hurtado, Ángel Martínez, Lluís Suñol i Chrístopher Tarí"
    },
    es: {
        intro_title: "Sobre ATLAS",
        intro_text: "ATLAS es un proyecto desarrollado en el Postgrado en Inteligencia Artificial aplicada al Transporte y la Logística de la UPC. Su objetivo es ayudar a un operador a decidir qué desvíos vale la pena incorporar entre un origen y un destino, maximizando la rentabilidad del trayecto sin comprometer capacidad ni desviar excesivamente la ruta.",
        hybrid_text: "El sistema analiza cientos de posibles paradas cercanas al corredor O→D. Para evaluarlas construimos un motor híbrido basado en tres pilares.",
        pillar1_text: "El primero es un módulo de modelos de Machine Learning que predice viajes diarios, beneficio y peso esperado para cada relación OD.",
        pillar2_text: "El segundo es un motor de scoring que normaliza y combina estas predicciones en una métrica única que favorece lo rentable y penaliza consumos de capacidad o desvíos altos.",
        pillar3_text: "El tercero es un algoritmo tipo mochila (greedy knapsack) que selecciona la combinación óptima de paradas respetando las restricciones del camión y el buffer.",
        optimization_text: "Una vez elegidas las mejores paradas, ATLAS reordena el itinerario y lo enruta sobre la red de carreteras usando PostGIS + pgRouting, generando varias alternativas reales O→D con geometría, tiempo, distancia y detalle económico.",
        transparency_text: "Un aspecto clave del proyecto es que todo el flujo es autoexplicativo y visual: cada etapa puede inspeccionarse sobre un mapa o un panel, permitiendo entender qué candidatos se evaluaron, qué score obtiene cada punto y cómo se construyen las rutas finales. Esto convierte ATLAS en una herramienta transparente, interpretable y alineada con escenarios logísticos reales.",
        
        footer_text: "El equipo de ATLAS está formado por Gabriel Bermúdez, Íngrid Hurtado, Ángel Martínez, Lluís Suñol y Chrístopher Tarí"
    },
    en: {
        intro_title: "About ATLAS",
        intro_text: "ATLAS is a project developed at the Postgraduate Program in Artificial Intelligence applied to Transport and Logistics at UPC. Its objective is to help an operator decide which detours are worth incorporating between an origin and destination, maximizing route profitability without compromising capacity or excessively deviating the route.",
        hybrid_text: "The system analyzes hundreds of possible stops near the O→D corridor. To evaluate them, we built a hybrid engine based on three pillars.",
        pillar1_text: "The first is a Machine Learning model module that predicts daily trips, profit and expected weight for each OD relationship.",
        pillar2_text: "The second is a scoring engine that normalizes and combines these predictions into a single metric that favors profitability and penalizes high capacity consumption or detours.",
        pillar3_text: "The third is a knapsack-type algorithm (greedy knapsack) that selects the optimal combination of stops respecting truck and buffer constraints.",
        optimization_text: "Once the best stops are selected, ATLAS reorders the itinerary and routes it over the road network using PostGIS + pgRouting, generating multiple real O→D alternatives with geometry, time, distance and economic detail.",
        transparency_text: "A key aspect of the project is that the entire flow is self-explanatory and visual: each stage can be inspected on a map or panel, allowing understanding of which candidates were evaluated, what score each point receives and how final routes are built. This makes ATLAS a transparent, interpretable tool aligned with real logistic scenarios.",
        
        footer_text: "The ATLAS team is formed by Gabriel Bermúdez, Íngrid Hurtado, Ángel Martínez, Lluís Suñol and Chrístopher Tarí"
    }
};

// Initialize language switcher
document.addEventListener('DOMContentLoaded', function() {
    // Get stored language or default to 'es'
    const storedLang = localStorage.getItem('selectedLanguage') || 'es';
    setLanguage(storedLang);
    
    // Add event listeners to language buttons
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const lang = this.getAttribute('data-lang');
            setLanguage(lang);
            localStorage.setItem('selectedLanguage', lang);
        });
    });
    
    // Print button
    document.getElementById('printBtn').addEventListener('click', function() {
        window.print();
    });
});

// Function to set language
function setLanguage(lang) {
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
            element.textContent = translations[lang][key];
        }
    });
    
    // Update language button states
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-lang') === lang) {
            btn.classList.add('active');
        }
    });
    
    // Update HTML lang attribute
    document.documentElement.lang = lang;
}
