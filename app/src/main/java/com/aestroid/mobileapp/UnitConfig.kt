package com.aestroid.mobileapp

import android.content.Context
import android.content.SharedPreferences

object UnitConfig {
    private const val PREFS_NAME = "unit_config"
    private const val KEY_UNIT_ID = "unit_id"
    private const val KEY_UNIT_TYPE = "unit_type"
    private const val DEFAULT_UNIT_ID = "UNIT-001"
    private const val DEFAULT_UNIT_TYPE = "mobile"

    private fun getSharedPreferences(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun getUnitId(context: Context): String {
        return getSharedPreferences(context).getString(KEY_UNIT_ID, DEFAULT_UNIT_ID) ?: DEFAULT_UNIT_ID
    }

    fun getUnitType(context: Context): String {
        return getSharedPreferences(context).getString(KEY_UNIT_TYPE, DEFAULT_UNIT_TYPE) ?: DEFAULT_UNIT_TYPE
    }

    fun setUnitId(context: Context, unitId: String) {
        getSharedPreferences(context).edit()
            .putString(KEY_UNIT_ID, unitId)
            .apply()
    }

    fun setUnitType(context: Context, unitType: String) {
        getSharedPreferences(context).edit()
            .putString(KEY_UNIT_TYPE, unitType)
            .apply()
    }

    // Common unit types for reference
    object UnitTypes {
        const val TANK = "tank"
        const val HELICOPTER = "helicopter"
        const val SOLDIER = "soldier"
        const val DRONE = "drone"
        const val VEHICLE = "vehicle"
        const val MOBILE = "mobile"
    }
}

