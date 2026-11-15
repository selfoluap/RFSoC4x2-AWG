import { useState } from "react";
import { DevicePanel } from "./components/DevicePanel";
import { Layout } from "./components/Layout";
import { LiveAdcPanel } from "./components/LiveAdcPanel";
import type { DeviceInfo } from "./device/types";
import { SignalGeneratorPage } from "./pages/SignalGeneratorPage";

export function App() {
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);

  return (
    <Layout>
      <DevicePanel onInfo={setDeviceInfo} />
      <SignalGeneratorPage />
      {deviceInfo?.capabilities.adc && <LiveAdcPanel />}
    </Layout>
  );
}
