import { useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { api } from "./api/client";
import { Layout } from "./components/Layout";
import { OverviewPage } from "./pages/OverviewPage";
import { ScenariosPage } from "./pages/ScenariosPage";
import { SegmentsPage } from "./pages/SegmentsPage";
import { AccuracyPage } from "./pages/AccuracyPage";

export function App() {
  const [snapshotDate, setSnapshotDate] = useState<string | undefined>();

  useEffect(() => {
    api
      .summary()
      .then((s) => setSnapshotDate(s.meta.snapshotDate))
      .catch(() => setSnapshotDate(undefined));
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout snapshotDate={snapshotDate} />}>
          <Route index element={<OverviewPage />} />
          <Route path="segments" element={<SegmentsPage />} />
          <Route path="scenarios" element={<ScenariosPage />} />
          <Route path="accuracy" element={<AccuracyPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
