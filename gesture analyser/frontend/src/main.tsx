import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

// NOT: React.StrictMode bilinçli olarak kullanılmıyor. StrictMode geliştirme
// modunda her efekti iki kez (setup→cleanup→setup) çalıştırır; bu, webcam
// (getUserMedia) ve WebSocket gibi imperatif kaynaklarda çift-açılış yarışına
// yol açıp streamReady'nin takılmasına ve bağlantı churn'üne neden oluyordu.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
