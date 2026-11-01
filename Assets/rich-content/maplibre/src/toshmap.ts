// The app's chat map: MapLibre on OpenFreeMap, on a style of its own or on PMTiles regions the app serves from its
// map folder under tiles/, as the server does for the web chat, with the answer's layers on top.
// scripts/build-map.sh builds it into toshmap.js.
import * as maplibregl from 'maplibre-gl';
import { Protocol } from 'pmtiles';
import { layers, namedFlavor } from '@protomaps/basemaps';

interface Layer {
	features: unknown[];
	drawn?: boolean;
	color: string;
}

interface Config {
	layers: Layer[];
	style?: string;
	regions: string[];
	/** Whether the map folder holds the fonts and icons the regions' labels need. */
	labels: boolean;
	dark: boolean;
	lang: string;
}

const OPEN_FREE_MAP = 'https://tiles.openfreemap.org/styles/liberty';
const ATTRIBUTION =
	'<a href="https://protomaps.com">Protomaps</a> © <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const host = window as any;

/** A plain ground, for a map with no base map or one that could not load. */
function ground(dark: boolean): maplibregl.StyleSpecification {
	return {
		layers: [{ id: 'ground', paint: { 'background-color': dark ? '#1f2937' : '#f3f4f6' }, type: 'background' }],
		sources: {},
		version: 8
	};
}

function baseStyle(config: Config): string | maplibregl.StyleSpecification {
	if (!config.regions.length) {
		if (config.style === 'none') return ground(config.dark);
		return config.style || OPEN_FREE_MAP;
	}

	const tiles = new URL('tiles/', location.href).href;
	const flavor = config.dark ? 'dark' : 'light';
	const sources: maplibregl.StyleSpecification['sources'] = {};
	const drawn: maplibregl.StyleSpecification['layers'] = [];

	config.regions.forEach((path, index) => {
		const name = `region${index}`;

		sources[name] = { attribution: ATTRIBUTION, type: 'vector', url: `pmtiles://${tiles}${encodeURI(path)}` };
		// one ground for all the regions, under every one of them
		for (const layer of layers(name, namedFlavor(flavor), { lang: config.lang })) {
			if (index > 0 && layer.type === 'background') continue;
			if (!config.labels && layer.type === 'symbol') continue;
			drawn.push({ ...layer, id: `${name}-${layer.id}` } as maplibregl.StyleSpecification['layers'][number]);
		}
	});

	if (!config.labels) return { layers: drawn, sources, version: 8 };

	return {
		glyphs: `${tiles}assets/fonts/{fontstack}/{range}.pbf`,
		layers: drawn,
		sources,
		sprite: `${tiles}assets/sprites/v4/${flavor}`,
		version: 8
	};
}

function bounds(config: Config): maplibregl.LngLatBoundsLike | null {
	let west = Infinity,
		south = Infinity,
		east = -Infinity,
		north = -Infinity;
	const visit = (value: unknown) => {
		if (!Array.isArray(value)) return;
		if (typeof value[0] === 'number' && typeof value[1] === 'number') {
			west = Math.min(west, value[0]);
			east = Math.max(east, value[0]);
			south = Math.min(south, value[1]);
			north = Math.max(north, value[1]);
		} else value.forEach(visit);
	};

	for (const layer of config.layers) {
		for (const feature of layer.features as { geometry?: { coordinates?: unknown } }[]) {
			visit(feature.geometry?.coordinates);
		}
	}

	return west === Infinity ? null : [[west, south], [east, north]];
}

let map: maplibregl.Map | null = null;

function render(config: Config): void {
	maplibregl.setWorkerUrl(URL.createObjectURL(new Blob([host.TOSH_MAP_WORKER], { type: 'text/javascript' })));

	maplibregl.addProtocol('pmtiles', new Protocol().tile);

	const created = new maplibregl.Map({
		attributionControl: { compact: true },
		container: 'map',
		scrollZoom: false,
		style: baseStyle(config)
	});

	map = created;
	created.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

	// without internet and without a region the base map cannot load: the layers still show on a plain ground
	let fellBack = false;

	created.on('error', () => {
		if (fellBack || created.isStyleLoaded()) return;
		fellBack = true;
		created.setStyle(ground(config.dark));
	});

	created.on('style.load', () => {
		config.layers.forEach((layer, index) => {
			const id = `tosh-${index}`;

			if (created.getSource(id)) return;

			created.addSource(id, { data: { features: layer.features, type: 'FeatureCollection' } as never, type: 'geojson' });
			created.addLayer({
				filter: ['in', ['geometry-type'], ['literal', ['Polygon', 'MultiPolygon']]] as never,
				id: `${id}-fill`,
				paint: { 'fill-color': layer.color, 'fill-opacity': 0.15 },
				source: id,
				type: 'fill'
			});
			created.addLayer({
				filter: ['in', ['geometry-type'], ['literal', ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']]] as never,
				id: `${id}-line`,
				paint: { 'line-color': layer.color, 'line-width': 2, ...(layer.drawn ? { 'line-dasharray': [3, 2] } : {}) },
				source: id,
				type: 'line'
			});
			created.addLayer({
				filter: ['in', ['geometry-type'], ['literal', ['Point', 'MultiPoint']]] as never,
				id: `${id}-point`,
				paint: {
					'circle-color': layer.color,
					'circle-opacity': 0.85,
					'circle-radius': 5,
					'circle-stroke-color': '#ffffff',
					'circle-stroke-width': 1
				},
				source: id,
				type: 'circle'
			});

			for (const part of ['fill', 'line', 'point']) {
				created.on('click', `${id}-${part}`, (event) => {
					const feature = event.features?.[0];

					if (feature) host.webkit.messageHandlers.select.postMessage({ layer: index, properties: feature.properties ?? {} });
				});
				created.on('mouseenter', `${id}-${part}`, () => (created.getCanvas().style.cursor = 'pointer'));
				created.on('mouseleave', `${id}-${part}`, () => (created.getCanvas().style.cursor = ''));
			}
		});
	});

	const box = bounds(config);

	if (box) created.fitBounds(box, { duration: 0, maxZoom: 17, padding: 24 });
	else created.jumpTo({ center: [0, 0], zoom: 1 });
}

host.ToshMap = { render, resize: () => map?.resize() };
