import { useEffect, useState } from 'react';
export default function Schemes(){
  const [country,setCountry]=useState('IN');
  const [state,setState]=useState('Maharashtra');
  const [district,setDistrict]=useState('');
  const [crop,setCrop]=useState('wheat');
  const [land,setLand]=useState(1.0);
  const [income,setIncome]=useState(200000);
  const [category,setCategory]=useState('small');
  const [irrigation,setIrrigation]=useState('rainfed');
  const [organic,setOrganic]=useState(false);
  const [benefit,setBenefit]=useState('');
  const [items,setItems]=useState([]);

  const go=async()=>{
    const u=new URL('/api/schemes', window.location.origin);
    if(country) u.searchParams.set('country',country);
    if(state)   u.searchParams.set('state',state);
    if(district)u.searchParams.set('district',district);
    if(crop)    u.searchParams.set('crop',crop);
    u.searchParams.set('land_size_ha',land);
    u.searchParams.set('annual_income',income);
    if(category)u.searchParams.set('category',category);
    if(irrigation)u.searchParams.set('irrigation',irrigation);
    if(organic) u.searchParams.set('organic','true');
    if(benefit) u.searchParams.set('benefit_preference',benefit);
    const r=await fetch(u); setItems(await r.json());
  };

  useEffect(()=>{go();},[]);

  return (
    <main style={{padding:24}}>
      <h2>Gov Schemes (Personalized)</h2>
      <div style={{display:'grid',gridTemplateColumns:'repeat(5, 220px)',gap:8,alignItems:'center'}}>
        <input value={country}  onChange={e=>setCountry(e.target.value)}  placeholder="country (e.g., IN)" />
        <input value={state}    onChange={e=>setState(e.target.value)}    placeholder="state" />
        <input value={district} onChange={e=>setDistrict(e.target.value)} placeholder="district (optional)" />
        <input value={crop}     onChange={e=>setCrop(e.target.value)}     placeholder="crop" />
        <select value={benefit} onChange={e=>setBenefit(e.target.value)}>
          <option value="">any benefit</option>
          <option value="subsidy">subsidy</option>
          <option value="credit">credit</option>
          <option value="insurance">insurance</option>
          <option value="equipment">equipment</option>
          <option value="training">training</option>
          <option value="market">market</option>
        </select>
        <input type="number" step="0.1" value={land} onChange={e=>setLand(e.target.value)} placeholder="land ha"/>
        <input type="number" step="1000" value={income} onChange={e=>setIncome(e.target.value)} placeholder="income INR"/>
        <input value={category} onChange={e=>setCategory(e.target.value)} placeholder="category"/>
        <input value={irrigation} onChange={e=>setIrrigation(e.target.value)} placeholder="irrigation"/>
        <label><input type="checkbox" checked={organic} onChange={e=>setOrganic(e.target.checked)} /> organic</label>
        <button onClick={go}>Search</button>
      </div>

      <ul>
        {items.map(x=>(
          <li key={x.id} style={{margin:'12px 0'}}>
            <b>{x.title}</b> <small>({x.benefit_type}, score {x.score.toFixed(2)})</small><br/>
            <span>{x.description}</span><br/>
            {x.url ? <a href={x.url} target="_blank" rel="noreferrer">{x.url}</a> : null}
          </li>
        ))}
      </ul>
    </main>
  );
}
