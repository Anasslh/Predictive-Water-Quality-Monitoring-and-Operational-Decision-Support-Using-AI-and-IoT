/*
  Warm Tier schema version 1.

  Run this script against the database selected in the connection string or
  with sqlcmd's -d option. It is safe to rerun: objects and indexes are created
  only when absent. Existing incompatible hand-created tables are rejected by
  warm_tier_etl.py rather than silently altered.
*/

SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.WarmSchemaVersion', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.WarmSchemaVersion
    (
        VersionNumber int NOT NULL
            CONSTRAINT PK_WarmSchemaVersion PRIMARY KEY,
        AppliedAt datetimeoffset(7) NOT NULL
            CONSTRAINT DF_WarmSchemaVersion_AppliedAt DEFAULT SYSUTCDATETIME(),
        Description nvarchar(256) NOT NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.Fact_WaterQuality', N'U') IS NOT NULL
AND (
       COL_LENGTH(N'dbo.Fact_WaterQuality', N'MeasurementId') IS NULL
    OR COL_LENGTH(N'dbo.Fact_WaterQuality', N'SiteId') IS NULL
    OR COL_LENGTH(N'dbo.Fact_WaterQuality', N'ParameterName') IS NULL
    OR COL_LENGTH(N'dbo.Fact_WaterQuality', N'MeasurementTimestamp') IS NULL
    OR COL_LENGTH(N'dbo.Fact_WaterQuality', N'SourceRecordHash') IS NULL
)
BEGIN
    THROW 50001,
        'Existing dbo.Fact_WaterQuality is incompatible with Warm schema v1; review and migrate it explicitly.',
        1;
END;
GO

IF OBJECT_ID(N'dbo.Fact_WaterQuality', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.Fact_WaterQuality
    (
        MeasurementId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_Fact_WaterQuality PRIMARY KEY,
        SiteId nvarchar(128) NOT NULL,
        ParameterName nvarchar(128) NOT NULL,
        Unit nvarchar(64) NULL,
        MeasurementTimestamp datetimeoffset(7) NOT NULL,
        PredictedValue float NULL,
        ActualValue float NULL,
        ShapTopFeatures nvarchar(max) NULL,
        IsAnomaly bit NULL,
        AnomalyScore float NULL,
        RetrainAlert nvarchar(2048) NULL,
        ForecastJson nvarchar(max) NULL,
        ModelVersion nvarchar(256) NULL,
        SourceFile nvarchar(512) NOT NULL,
        SourceRecordHash binary(32) NOT NULL,
        IngestedAt datetimeoffset(7) NOT NULL
            CONSTRAINT DF_Fact_WaterQuality_IngestedAt DEFAULT SYSUTCDATETIME(),

        CONSTRAINT UQ_Fact_WaterQuality_Record
            UNIQUE (SiteId, ParameterName, MeasurementTimestamp),
        CONSTRAINT CK_Fact_WaterQuality_ParameterName
            CHECK (LEN(LTRIM(RTRIM(ParameterName))) > 0),
        CONSTRAINT CK_Fact_WaterQuality_SiteId
            CHECK (LEN(LTRIM(RTRIM(SiteId))) > 0),
        CONSTRAINT CK_Fact_WaterQuality_ShapJson
            CHECK (ShapTopFeatures IS NULL OR ISJSON(ShapTopFeatures) = 1),
        CONSTRAINT CK_Fact_WaterQuality_ForecastJson
            CHECK (ForecastJson IS NULL OR ISJSON(ForecastJson) = 1)
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.Fact_WaterQuality')
      AND name = N'IX_Fact_WaterQuality_ParameterTime'
)
BEGIN
    CREATE INDEX IX_Fact_WaterQuality_ParameterTime
        ON dbo.Fact_WaterQuality (ParameterName, MeasurementTimestamp DESC)
        INCLUDE (SiteId, Unit, ActualValue, PredictedValue, IsAnomaly);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.Fact_WaterQuality')
      AND name = N'IX_Fact_WaterQuality_SiteParameterTime'
)
BEGIN
    CREATE INDEX IX_Fact_WaterQuality_SiteParameterTime
        ON dbo.Fact_WaterQuality (SiteId, ParameterName, MeasurementTimestamp DESC)
        INCLUDE (Unit, ActualValue, PredictedValue, IsAnomaly, AnomalyScore);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.Fact_WaterQuality')
      AND name = N'IX_Fact_WaterQuality_Anomalies'
)
BEGIN
    CREATE INDEX IX_Fact_WaterQuality_Anomalies
        ON dbo.Fact_WaterQuality (SiteId, ParameterName, MeasurementTimestamp DESC)
        INCLUDE (ActualValue, PredictedValue, AnomalyScore)
        WHERE IsAnomaly = 1;
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM dbo.WarmSchemaVersion WHERE VersionNumber = 1
)
BEGIN
    INSERT dbo.WarmSchemaVersion (VersionNumber, Description)
    VALUES (1, N'Initial generic Warm measurement schema');
END;
GO
