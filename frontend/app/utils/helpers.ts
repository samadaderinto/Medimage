export function getRandomInt(min: number, max: number) {
  min = Math.ceil(min); // Round up to ensure inclusive minimum
  max = Math.floor(max); // Round down to ensure exclusive maximum
  return Math.floor(Math.random() * (max - min) + min);
}
