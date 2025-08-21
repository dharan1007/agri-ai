import { useState } from 'react';
export default function Calendar(){
  const [lat,setLat]=useState(19.07);
  const [lon,setLon]=useState(72.88);
  const [crop,setCrop]=useState('wheat');
  const [data,setData]=useState(null);
  const go=async()=>{
    const u=new URL('/api/calendar', window.location.origin);
    u.searchParams.set('lat',lat);
    u.searchParams.set('lon',lon);
    u.searchParams.set('crop',crop);
    const r=await fetch(u); setData(await r.json());
  };
  return (
    <main style={{padding:24}}>
      <h2>Crop Calendar (Live Weather)</h2>
      <div>
        <input value={lat} onChange={e=>setLat(e.target.value)} />
        <input value={lon} onChange={e=>setLon(e.target.value)} />
        <input value={crop} onChange={e=>setCrop(e.target.value)} />
        <button onClick={go}>Get</button>
      </div>
      {data && <pre>{JSON.stringify(data,null,2)}</pre>}
    </main>
  );
}
