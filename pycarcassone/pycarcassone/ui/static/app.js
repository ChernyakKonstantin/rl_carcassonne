const TILE_SIZE = 92;
let state = null;
let selectedOrientation = 0;
let selectedPosition = null;
let selectedMeeple = null;

const propertyNames = {
  FIELD: "Field",
  ROAD: "Road",
  CITY: "City",
  ABBOT: "Abbot"
};
const hatchablePropertyTypes = new Set(["FIELD", "ROAD", "CITY"]);

async function fetchState() {
  const response = await fetch("/api/state");
  state = await response.json();
  normalizeSelection();
  render();
}

async function newGame() {
  const seed = Number(document.getElementById("seed").value || 67);
  const nOpponents = Number(document.getElementById("opponents").value || 2);
  const response = await fetch("/api/new", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({seed, n_opponents: nOpponents})
  });
  state = await response.json();
  selectedOrientation = 0;
  selectedPosition = null;
  selectedMeeple = null;
  normalizeSelection();
  render();
}

async function placeAction() {
  const action = selectedAction();
  if (!action) {
    return;
  }
  const response = await fetch("/api/action", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({action_index: action.index})
  });
  state = await response.json();
  selectedPosition = null;
  selectedMeeple = null;
  normalizeSelection();
  render();
}

function normalizeSelection() {
  if (!state || !state.current_turn) {
    return;
  }
  const orientations = [...new Set(state.current_turn.actions.map(action => action.orientation))];
  if (!orientations.includes(selectedOrientation)) {
    selectedOrientation = orientations[0] ?? 0;
  }
}

function render() {
  renderStatus();
  renderPlayers();
  renderCurrentCard();
  renderBoard();
}

function renderStatus() {
  document.getElementById("status").textContent = state.message;
  document.getElementById("deck").textContent = `Deck: ${state.deck_remaining}`;
  const selection = document.getElementById("selection");
  if (!selectedPosition) {
    selection.textContent = "";
    return;
  }
  const meepleText = selectedMeeple === null ? "no meeple" : `property ${selectedMeeple}`;
  selection.textContent = `Selected: (${selectedPosition.y}, ${selectedPosition.x}), ${selectedOrientation * 90} deg, ${meepleText}`;
}

function renderPlayers() {
  const container = document.getElementById("players");
  container.replaceChildren();
  for (const player of state.players) {
    const row = document.createElement("div");
    row.className = "player";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = player.color;
    const name = document.createElement("strong");
    name.textContent = player.label;
    const stats = document.createElement("span");
    stats.className = "player-stats";
    stats.textContent = `${player.score} pts / ${player.remaining_meeples} meeples`;
    row.append(swatch, name, stats);
    container.append(row);
  }
}

function renderCurrentCard() {
  const cardContainer = document.getElementById("current-card");
  const meepleControls = document.getElementById("meeple-controls");
  const placeButton = document.getElementById("place-action");
  cardContainer.replaceChildren();
  meepleControls.replaceChildren();

  if (!state.current_turn) {
    cardContainer.textContent = "Game over";
    placeButton.disabled = true;
    return;
  }

  const cardOption = currentCardOption();
  for (const option of state.current_turn.card.options) {
    const actions = state.current_turn.actions.filter(action => action.orientation === option.orientation);
    const placementCount = new Set(actions.map(action => `${action.position.y}:${action.position.x}`)).size;
    const preview = document.createElement("button");
    preview.type = "button";
    preview.className = "orientation-preview";
    preview.title = `${placementCount} placements, ${actions.length} action variants`;
    preview.disabled = actions.length === 0;
    if (option.orientation === selectedOrientation) {
      preview.classList.add("active");
    }
    preview.append(renderTile(option.values, option.properties, propertyDataForCardOption(option), true, "mini"));
    const caption = document.createElement("span");
    caption.className = "orientation-caption";
    caption.textContent = `${option.angle} deg (${placementCount})`;
    preview.append(caption);
    preview.addEventListener("click", () => {
      selectedOrientation = option.orientation;
      selectedPosition = null;
      selectedMeeple = null;
      render();
    });
    cardContainer.append(preview);
  }

  if (selectedPosition) {
    const actions = actionsForSelectedPosition();
    const none = actions.find(action => action.meeple_position === null);
    if (none) {
      meepleControls.append(meepleButton("No meeple", null));
    }
    for (const action of actions.filter(action => action.meeple_position !== null)) {
      const propertyInfo = cardOption.property_types.find(item => item.index === action.meeple_position);
      const label = `${propertyNames[propertyInfo?.type] ?? "Property"} ${action.meeple_position}`;
      meepleControls.append(meepleButton(label, action.meeple_position));
    }
  }

  placeButton.disabled = !selectedAction();
}

function meepleButton(label, meeplePosition) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.className = selectedMeeple === meeplePosition ? "active" : "";
  button.addEventListener("click", () => {
    selectedMeeple = meeplePosition;
    render();
  });
  return button;
}

function renderBoard() {
  const board = document.getElementById("board");
  board.replaceChildren();
  const legalPositions = legalPositionsForOrientation();
  const positions = [
    ...state.board.tiles.map(tile => tile.position),
    ...legalPositions
  ];
  const bounds = getBounds(positions);
  const width = (bounds.maxX - bounds.minX + 1) * TILE_SIZE;
  const height = (bounds.maxY - bounds.minY + 1) * TILE_SIZE;
  board.style.width = `${Math.max(width, 320)}px`;
  board.style.height = `${Math.max(height, 320)}px`;

  for (const tile of state.board.tiles) {
    const element = renderTile(tile.values, tile.properties, tile.property_data, false);
    placeAt(element, tile.position, bounds);
    board.append(element);
  }

  for (const position of legalPositions) {
    const overlay = document.createElement("div");
    overlay.className = "placement";
    if (selectedPosition && selectedPosition.y === position.y && selectedPosition.x === position.x) {
      overlay.classList.add("selected");
    }
    const plus = document.createElement("div");
    plus.className = "plus";
    plus.textContent = "+";
    overlay.append(plus);
    overlay.addEventListener("click", () => {
      selectedPosition = {y: position.y, x: position.x};
      selectedMeeple = null;
      render();
    });
    placeAt(overlay, position, bounds);
    board.append(overlay);
  }
}

function renderTile(values, properties, propertyData, showPropertyLabels, size = "normal") {
  const tile = document.createElement("div");
  tile.className = size === "mini" ? "tile mini" : showPropertyLabels ? "tile small" : "tile";
  const propertyByIndex = propertyDataByIndex(propertyData);
  const playerById = Object.fromEntries(state.players.map(player => [player.id, player]));
  for (let cellIndex = 0; cellIndex < values.length; cellIndex += 1) {
    const symbol = values[cellIndex];
    const propertyIndex = properties[cellIndex];
    const property = propertyByIndex.get(propertyIndex);
    const cell = document.createElement("div");
    cell.className = `cell ${symbol}`;
    const ownerColors = ownerColorsForProperty(property, playerById);
    if (ownerColors.length > 0) {
      cell.style.backgroundImage = ownershipPattern(ownerColors);
    }
    tile.append(cell);
  }

  const centers = propertyCenters(properties);
  const occupiedCenters = [];
  if (showPropertyLabels) {
    for (const [propertyIndex, center] of centers.entries()) {
      const label = document.createElement("span");
      label.className = "property-label";
      label.style.left = `${center.x}%`;
      label.style.top = `${center.y}%`;
      label.textContent = propertyIndex;
      tile.append(label);
      occupiedCenters.push(center);
    }
  }

  for (const property of propertyData) {
    if (property.owner && !property.ignored) {
      const center = centers.get(property.index);
      if (center) {
        occupiedCenters.push(center);
      }
    }
  }

  for (const property of propertyData) {
    if (!property.shield || property.ignored) {
      continue;
    }
    const center = shieldCenter(properties, property.index, centers.get(property.index), occupiedCenters);
    if (!center) {
      continue;
    }
    const shield = document.createElement("span");
    shield.className = "shield";
    shield.style.left = `${center.x}%`;
    shield.style.top = `${center.y}%`;
    shield.textContent = "S";
    tile.append(shield);
    occupiedCenters.push(center);
  }

  for (const property of propertyData) {
    if (!property.owner || property.ignored) {
      continue;
    }
    const center = centers.get(property.index);
    const player = playerById[property.owner];
    if (!center || !player) {
      continue;
    }
    const meeple = document.createElement("span");
    meeple.className = "meeple";
    meeple.style.left = `${center.x}%`;
    meeple.style.top = `${center.y}%`;
    meeple.style.background = player.color;
    meeple.textContent = player.label === "You" ? "Y" : player.label.replace("P", "");
    tile.append(meeple);
  }
  return tile;
}

function propertyDataForCardOption(option) {
  return option.property_types.map(property => ({
    index: property.index,
    type: property.type,
    type_value: property.type_value,
    owner: null,
    owners: [],
    ignored: false,
    shield: Boolean(property.shield)
  }));
}

function propertyDataByIndex(propertyData) {
  return new Map(propertyData.map(property => [property.index, property]));
}

function ownerColorsForProperty(property, playerById) {
  if (!property || property.ignored || !hatchablePropertyTypes.has(property.type)) {
    return [];
  }
  const ownerIds = property.owners ?? (property.owner ? [property.owner] : []);
  const uniqueOwnerIds = [...new Set(ownerIds)];
  return uniqueOwnerIds
    .map(ownerId => playerById[ownerId]?.color)
    .filter(color => color);
}

function ownershipPattern(colors) {
  const band = 2;
  const gap = 5;
  const stops = [];
  let cursor = 0;
  for (const color of colors) {
    stops.push(`transparent ${cursor}px ${cursor + gap}px`);
    stops.push(`${hexToRgba(color, 0.62)} ${cursor + gap}px ${cursor + gap + band}px`);
    cursor += gap + band;
  }
  stops.push(`transparent ${cursor}px ${cursor + gap}px`);
  return `repeating-linear-gradient(135deg, ${stops.join(", ")})`;
}

function hexToRgba(hexColor, alpha) {
  const normalized = hexColor.replace("#", "");
  const red = parseInt(normalized.slice(0, 2), 16);
  const green = parseInt(normalized.slice(2, 4), 16);
  const blue = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function propertyCenters(properties) {
  const sums = new Map();
  properties.forEach((propertyIndex, cellIndex) => {
    if (propertyIndex === null || propertyIndex === undefined) {
      return;
    }
    const row = Math.floor(cellIndex / 5);
    const col = cellIndex % 5;
    if (!sums.has(propertyIndex)) {
      sums.set(propertyIndex, {x: 0, y: 0, n: 0});
    }
    const item = sums.get(propertyIndex);
    item.x += (col + 0.5) * 20;
    item.y += (row + 0.5) * 20;
    item.n += 1;
  });
  const centers = new Map();
  for (const [propertyIndex, item] of sums.entries()) {
    centers.set(propertyIndex, {x: item.x / item.n, y: item.y / item.n});
  }
  return centers;
}

function propertyCells(properties) {
  const cells = new Map();
  properties.forEach((propertyIndex, cellIndex) => {
    if (propertyIndex === null || propertyIndex === undefined) {
      return;
    }
    const row = Math.floor(cellIndex / 5);
    const col = cellIndex % 5;
    if (!cells.has(propertyIndex)) {
      cells.set(propertyIndex, []);
    }
    cells.get(propertyIndex).push({
      x: (col + 0.5) * 20,
      y: (row + 0.5) * 20
    });
  });
  return cells;
}

function shieldCenter(properties, propertyIndex, preferredCenter, occupiedCenters) {
  const cells = propertyCells(properties);
  const candidates = cells.get(propertyIndex) ?? [];
  if (candidates.length === 0) {
    return preferredCenter;
  }
  if (occupiedCenters.length === 0) {
    return closestPoint(candidates, preferredCenter);
  }

  let best = candidates[0];
  let bestFreeDistance = Number.NEGATIVE_INFINITY;
  let bestPreferredDistance = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    let nearestOccupiedDistance = Number.POSITIVE_INFINITY;
    for (const occupiedCenter of occupiedCenters) {
      const distance = squaredDistance(candidate, occupiedCenter);
      nearestOccupiedDistance = Math.min(nearestOccupiedDistance, distance);
    }
    const preferredDistance = preferredCenter ? squaredDistance(candidate, preferredCenter) : 0;
    if (
      nearestOccupiedDistance > bestFreeDistance ||
      (nearestOccupiedDistance === bestFreeDistance && preferredDistance < bestPreferredDistance)
    ) {
      best = candidate;
      bestFreeDistance = nearestOccupiedDistance;
      bestPreferredDistance = preferredDistance;
    }
  }
  return best;
}

function closestPoint(points, target) {
  if (!target) {
    return points[0];
  }
  let best = points[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const point of points) {
    const distance = squaredDistance(point, target);
    if (distance < bestDistance) {
      best = point;
      bestDistance = distance;
    }
  }
  return best;
}

function squaredDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function currentCardOption() {
  return state.current_turn.card.options.find(option => option.orientation === selectedOrientation);
}

function legalPositionsForOrientation() {
  if (!state.current_turn) {
    return [];
  }
  const seen = new Set();
  const positions = [];
  for (const action of state.current_turn.actions) {
    if (action.orientation !== selectedOrientation) {
      continue;
    }
    const key = `${action.position.y}:${action.position.x}`;
    if (!seen.has(key)) {
      seen.add(key);
      positions.push(action.position);
    }
  }
  return positions;
}

function actionsForSelectedPosition() {
  if (!selectedPosition || !state.current_turn) {
    return [];
  }
  return state.current_turn.actions.filter(action =>
    action.orientation === selectedOrientation &&
    action.position.y === selectedPosition.y &&
    action.position.x === selectedPosition.x
  );
}

function selectedAction() {
  return actionsForSelectedPosition().find(action => action.meeple_position === selectedMeeple);
}

function getBounds(positions) {
  if (positions.length === 0) {
    return {minX: 0, maxX: 0, minY: 0, maxY: 0};
  }
  return {
    minX: Math.min(...positions.map(position => position.x)),
    maxX: Math.max(...positions.map(position => position.x)),
    minY: Math.min(...positions.map(position => position.y)),
    maxY: Math.max(...positions.map(position => position.y))
  };
}

function placeAt(element, position, bounds) {
  element.style.left = `${(position.x - bounds.minX) * TILE_SIZE}px`;
  element.style.top = `${(position.y - bounds.minY) * TILE_SIZE}px`;
}

document.getElementById("new-game").addEventListener("click", newGame);
document.getElementById("place-action").addEventListener("click", placeAction);
fetchState();
