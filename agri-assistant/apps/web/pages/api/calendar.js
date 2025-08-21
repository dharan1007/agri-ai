export default async function handler(req, res) {
  const u = new URL('http://crop-calendar:8000/calendar');
  for (const [k,v] of Object.entries(req.query)) if (v != null) u.searchParams.set(k, v);
  try {
    const r = await fetch(u.toString());
    const text = await r.text();
    res.status(r.status).send(text);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
}
