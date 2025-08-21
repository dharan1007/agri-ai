import { useState } from 'react';
export default function Predict(){
  const [file,setFile]=useState(null);
  const [resp,setResp]=useState(null);
  const onSubmit=async(e)=>{
    e.preventDefault();
    if(!file) return;
    const fd=new FormData();
    fd.append('file',file);
    const r=await fetch('/api/predict',{method:'POST',body:fd});
    const data = await r.json();
    setResp(data);
  };
  return (
    <main style={{padding:24}}>
      <h2>Disease Predictor</h2>
      <form onSubmit={onSubmit}>
        <input type="file" accept="image/*" onChange={e=>setFile(e.target.files[0])}/>
        <button type="submit">Predict</button>
      </form>
      {resp && <pre>{JSON.stringify(resp,null,2)}</pre>}
    </main>
  );
}
