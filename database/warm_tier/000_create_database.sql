/*
  Provision the Warm-tier database with sqlcmd variable substitution.

  Example:
    sqlcmd -S localhost\SQLEXPRESS -E -C \
      -v DatabaseName="NileWaterQuality_Warm" \
      -i database\warm_tier\000_create_database.sql

  DatabaseName must be supplied with ``sqlcmd -v``. This script is idempotent
  and does not alter an existing database.
*/

IF DB_ID(N'$(DatabaseName)') IS NULL
BEGIN
    DECLARE @create_database nvarchar(max) =
        N'CREATE DATABASE ' + QUOTENAME(N'$(DatabaseName)') + N';';
    EXEC sys.sp_executesql @create_database;
END;
GO
