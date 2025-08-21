export const config = { api: { bodyParser: false } };

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  try {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = Buffer.concat(chunks);

    const r = await fetch('http://disease-predictor:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': req.headers['content-type'] || 'application/octet-stream' },
      body
    });

    const text = await r.text();
    res.status(r.status).send(text);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
}
