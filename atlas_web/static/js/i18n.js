// Multilingual translation data
const translations = {
    ca: {
        intro_title: "Sobre ATLAS",
        intro_text: "<a href='https://atlasproject.duckdns.org' target='_blank'>ATLAS</a> és un projecte desenvolupat al Postgrau en Intel·ligència Artificial aplicada al Transport i la Logística de la UPC. <b>El seu objectiu és ajudar un operador a decidir quins desviaments val la pena incorporar</b> en una ruta entre un origen i una destinació, <b>maximitzant la rendibilitat</b> del trajecte sense comprometre capacitat ni desviar excessivament la ruta.",
        hybrid_text: "El sistema analitza <b>centenars de possibles parades</b> properes al corredor origen-destinació. Per a avaluar-les construïm un <b>motor híbrid basat en tres pilars</b>.",
        pillar1_text: "El primer és un mòdul de models de <b>Machine Learning basat en XGBoost</b>, un algoritme que utilitza arbres de decisió. Entrenat amb més de 20 milions de registres de dades de mobilitat, <b>prediu viatges diaris, benefici i pes</b> esperat per a cada relació origen-destinació.",
        pillar2_text: "El segon és un <b>motor de scoring</b> que normalitza i combina aquestes prediccions en una mètrica única que afavoreix la rendibilitat i penalitza consums de capacitat o desviaments alts.",
        pillar3_text: "El tercer és un <b>algoritme tipus motxilla (greedy knapsack)</b> que selecciona la combinació òptima de parades respectant les restriccions del camió i el desvío màxim permès.",
        optimization_text: "Una vegada escollides les millors parades, ATLAS reordena l'itinerari i <b>l'enruta sobre la xarxa de carreteres</b>, generant <b>diverses alternatives reals</b> origen-destinació amb geometria, temps, distància i detall econòmic.",
        transparency_text: "Un aspecte clau del projecte és que <b>tot el flux és autoexplicatiu i visual</b>: cada etapa pot inspeccionarse al mapa o al panell de resultats, permetent entendre quins candidats es van avaluar, quin score obté cada punt i com es construeixen les rutes finals. Això converteix ATLAS en una eina <b>transparent, interpretable i alineada amb escenaris logístics reals</b>.",
        
        footer_text: "L'equip d'<a href='https://atlasproject.duckdns.org' target='_blank'>ATLAS</a> està format per <a href='https://www.linkedin.com/in/gabrielbermudez/' target='_blank'>Gabriel Bermúdez</a>, <a href='https://www.linkedin.com/in/ingridhurtadocruz/' target='_blank'>Íngrid Hurtado</a>, <a href='https://www.linkedin.com/in/angelmp/' target='_blank'>Ángel Martínez</a>, <a href='https://www.linkedin.com/in/lluissunol' target='_blank'>Lluís Suñol</a> i Chrístopher Tarí"
    },
    es: {
        intro_title: "Sobre ATLAS",
        intro_text: "<a href='https://atlasproject.duckdns.org' target='_blank'>ATLAS</a> es un proyecto desarrollado en el Postgrado en Inteligencia Artificial aplicada al Transporte y la Logística de la UPC. <b>Su objetivo es ayudar a un operador a decidir qué desvíos vale la pena incorporar</b> en una ruta entre un origen y un destino, <b>maximizando la rentabilidad</b> del trayecto sin comprometer capacidad ni desviar excesivamente la ruta.",
        hybrid_text: "El sistema analiza cientos de posibles paradas cercanas al corredor origen-destino. Para evaluarlas construimos un <b>motor híbrido basado en tres pilares</b>.",
        pillar1_text: "El primero es un módulo de modelos de <b>Machine Learning basado en XGBoost</b>, un algoritmo que utiliza árboles de decisión. Entrenado con más de 20 millones de registros de datos de movilidad, <b>predice viajes diarios, beneficio y peso</b> esperado para cada relación origen-destino.",
        pillar2_text: "El segundo es un <b>motor de scoring</b> que normaliza y combina estas predicciones en una métrica única que favorece la rentabilidad y penaliza consumos de capacidad o desvíos altos.",
        pillar3_text: "El tercero es un <b>algoritmo tipo mochila (greedy knapsack)</b> que selecciona la combinación óptima de paradas respetando las restricciones del camión y el desvío máximo permitido.",
        optimization_text: "Una vez elegidas las mejores paradas, ATLAS reordena el itinerario y <b>lo enruta sobre la red de carreteras</b>, generando <b>varias alternativas reales</b> origen-destino con geometría, tiempo, distancia y detalle económico.",
        transparency_text: "Un aspecto clave del proyecto es que <b>todo el flujo es autoexplicativo y visual</b>: cada etapa puede inspeccionarse en el mapa o en el panel de resultados, permitiendo entender qué candidatos se evaluaron, qué score obtiene cada punto y cómo se construyen las rutas finales. Esto convierte ATLAS en una herramienta <b>transparente, interpretable y alineada con escenarios logísticos reales</b>.",
        
        footer_text: "El equipo de <a href='https://atlasproject.duckdns.org' target='_blank'>ATLAS</a> está formado por <a href='https://www.linkedin.com/in/gabrielbermudez/' target='_blank'>Gabriel Bermúdez</a>, <a href='https://www.linkedin.com/in/ingridhurtadocruz/' target='_blank'>Íngrid Hurtado</a>, <a href='https://www.linkedin.com/in/angelmp/' target='_blank'>Ángel Martínez</a>, <a href='https://www.linkedin.com/in/lluissunol' target='_blank'>Lluís Suñol</a> y Chrístopher Tarí"
    },
    en: {
        intro_title: "About ATLAS",
        intro_text: "<a href='https://atlasproject.duckdns.org' target='_blank'>ATLAS</a> is a project developed at the Postgraduate Program in Artificial Intelligence applied to Transport and Logistics at UPC. <b>Its objective is to help an operator decide which detours are worth incorporating</b> in a route between an origin and destination, <b>maximizing route profitability</b> without compromising capacity or excessively deviating the route.",
        hybrid_text: "The system analyzes hundreds of possible stops near the origin-destination corridor. To evaluate them, we built a <b>hybrid engine based on three pillars</b>.",
        pillar1_text: "The first is a <b>Machine Learning model module based on XGBoost</b>, an algorithm that uses decision trees. Trained with over 20 million mobility data records, <b>it predicts daily trips, profit and expected weight</b> for each origin-destination relationship.",
        pillar2_text: "The second is a <b>scoring engine</b> that normalizes and combines these predictions into a single metric that favors profitability and penalizes high capacity consumption or detours.",
        pillar3_text: "The third is a <b>knapsack-type algorithm (greedy knapsack)</b> that selects the optimal combination of stops respecting truck constraints and maximum allowed detour.",
        optimization_text: "Once the best stops are selected, ATLAS reorders the itinerary and <b>routes it over the road network</b>, generating <b>multiple real alternatives</b> with geometry, time, distance and economic detail.",
        transparency_text: "A key aspect of the project is that <b>the entire flow is self-explanatory and visual</b>: each stage can be inspected on the map or results panel, allowing understanding of which candidates were evaluated, what score each point receives and how final routes are built. This makes ATLAS a <b>transparent, interpretable tool aligned with real logistic scenarios</b>.",
        
        footer_text: "The <a href='https://atlasproject.duckdns.org' target='_blank'>ATLAS</a> team is formed by <a href='https://www.linkedin.com/in/gabrielbermudez/' target='_blank'>Gabriel Bermúdez</a>, <a href='https://www.linkedin.com/in/ingridhurtadocruz/' target='_blank'>Íngrid Hurtado</a>, <a href='https://www.linkedin.com/in/angelmp/' target='_blank'>Ángel Martínez</a>, <a href='https://www.linkedin.com/in/lluissunol' target='_blank'>Lluís Suñol</a> and Chrístopher Tarí"
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
            element.innerHTML = translations[lang][key];
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
