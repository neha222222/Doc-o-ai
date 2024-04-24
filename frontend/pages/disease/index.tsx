import React from "react";
import Head from "next/head";
import ModelLayout from "../../layout/ModelLayout";
import AllDisease from "../../features/All_Models/AllDisease";
import HomeStyles from "../../styles/Home.module.css";
import { Box } from "@mui/material";

function Vehicle_testing() {
  return (
    <>
      <Head>
        <title>AI Doctor| Diseases </title>
        <meta name="description" content="Watcher Police help Web App" />
        <meta name="viewport" content="width=1200, minimum-scale=0.25" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Box className={HomeStyles.body}>
        <ModelLayout>
          <AllDisease />
        </ModelLayout>
      </Box>
    </>
  );
}

export default Vehicle_testing;
