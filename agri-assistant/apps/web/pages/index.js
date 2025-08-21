export default function Home(){
  return (
    <main style={{padding:24, fontFamily:'sans-serif'}}>
      <h1>Agricultural AI Assistant</h1>
      <ul>
        <li><a href="/predict">Disease Predictor</a></li>
        <li><a href="/schemes">Gov Schemes</a></li>
        <li><a href="/calendar">Crop Calendar</a></li>
      </ul>
    </main>
  );
}
