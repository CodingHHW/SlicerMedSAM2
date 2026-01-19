import io
import gzip
import requests
import copy
import threading
import time

import importlib.util

import numpy as np
from pathlib import Path

import slicer
import qt
import vtk
from qt import QApplication, QPalette

from vtkmodules.util.numpy_support import vtk_to_numpy

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


###############################################################################
# Decorators and utility functions
###############################################################################


DEBUG_MODE = False


def debug_print(*args):
    if DEBUG_MODE:
        print(*args)


def ensure_synched(func):
    """
    Decorator that ensures the image and segment are synced before calling
    the actual prompt function.
    """

    def inner(self, *args, **kwargs):
        failed_to_sync = False

        if self.image_changed():
            debug_print(
                "Image changed (or not previously set). Calling upload_segment_to_server()"
            )
            result = self.upload_image_to_server()

            failed_to_sync = result is None

        if not failed_to_sync and self.selected_segment_changed():
            debug_print(
                "Segment changed (or not previously set). Calling upload_segment_to_server()"
            )
            self.remove_all_but_last_prompt()
            result = self.upload_segment_to_server()

            failed_to_sync = result is None
        else:
            debug_print("Segment did not change!")

        if not failed_to_sync:
            return func(self, *args, **kwargs)

    return inner


###############################################################################
# MedSAM2
###############################################################################


class MedSAM2(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = _("MedSAM2")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Segmentation")
        ]
        self.parent.dependencies = []  # List other modules if needed
        self.parent.contributors = ["CodingHHW"]
        self.parent.helpText = """
            This is a 3D Slicer extension for using MedSAM2.

            Read more about this plugin here: https://github.com/CodingHHW/SlicerMedSAM2.
            """
        self.parent.acknowledgementText = """When using SlicerMedSAM2, please cite the relevant publications."""


###############################################################################
# MedSAM2Widget
###############################################################################


class MedSAM2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    ###############################################################################
    # Setup and initialization functions
    ###############################################################################

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self):
        """
        Overridden setup method. Initializes UI and setups up prompts.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.install_dependencies()

        ui_widget = slicer.util.loadUI(self.resourcePath("UI/MedSAM2.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # Set up editor widget
        self.ui.editor_widget.setMaximumNumberOfUndoStates(10)
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        # Use the same segmentation parameter node as the Segment Editor core module
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode")
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        self.ui.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.ui.editor_widget.setSegmentationNode(self.get_segmentation_node())

        # Set up style sheets for selected/unselected buttons
        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "PointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "button": self.ui.pbInteractionPoint,
                "button_text": self.ui.pbInteractionPoint.text,
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
                "button_icon_filename": "bbox_icon.svg",
            },
        }

        self.setup_shortcuts()

        self.all_prompt_buttons = {}
        self.setup_prompts()

        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

    def init_ui_functionality(self):
        """
        Connect UI elements to functions.
        """
        self.ui.uploadProgressGroup.setVisible(False)

        # Load the saved server URL (default to http://172.25.52.123:8002/ if not set)
        savedServer = slicer.util.settingsValue("MedSAM2/server", "http://172.25.52.123:8002")
        self.ui.Server.text = savedServer
        self.server = savedServer.rstrip("/")

        self.ui.Server.editingFinished.connect(self.update_server)
        self.ui.pbTestServer.clicked.connect(self.test_server_connection)

        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)

        # Connect Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(
            self.on_prompt_type_positive_clicked
        )
        self.ui.pbPromptTypeNegative.clicked.connect(
            self.on_prompt_type_negative_clicked
        )

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(), 
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)

    def setup_shortcuts(self):
        """
        Sets up keyboard shortcuts.
        """
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "t": self.toggle_prompt_type,  # Add 'T' shortcut to toggle between positive/negative
        }
        self.shortcut_items = {}

        for shortcut_key, shortcut_event in shortcuts.items():
            debug_print(f"Added shortcut for {shortcut_key}: {shortcut_event}")
            shortcut = qt.QShortcut(
                qt.QKeySequence(shortcut_key), slicer.util.mainWindow()
            )
            shortcut.activated.connect(shortcut_event)
            self.shortcut_items[shortcut_key] = shortcut

    def remove_shortcut_items(self):
        """
        Called at cleanup to remove all the shortcuts we attached.
        """
        if hasattr(self, "shortcut_items"):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()
                shortcut = None

    def install_dependencies(self):
        """
        Checks for (and installs if needed) python packages needed by the module.
        """
        dependencies = {
            "requests_toolbelt": "requests_toolbelt",
        }

        for dependency in dependencies:
            if self.check_dependency_installed(dependency, dependencies[dependency]):
                continue
            self.run_with_progress_bar(
                self.pip_install_wrapper,
                (dependencies[dependency],),
                "Installing dependencies: %s" % dependency,
            )

    def check_dependency_installed(self, import_name, module_name_and_version):
        """
        Checks if a package is installed with the correct version.
        """
        if "==" in module_name_and_version:
            module_name, module_version = module_name_and_version.split("==")
        else:
            module_name = module_name_and_version
            module_version = None

        spec = importlib.util.find_spec(import_name)
        if spec is None:
            # Not installed
            return False

        if module_version is not None:
            import importlib.metadata as metadata
            try:
                version = metadata.version(module_name)
                if version != module_version:
                    # Version mismatch
                    return False
            except metadata.PackageNotFoundError:
                debug_print(f"Could not determine version for {module_name}.")

        return True

    def pip_install_wrapper(self, command, event):
        """
        Installs pip packages.
        """
        slicer.util.pip_install(command)
        event.set()

    def run_with_progress_bar(self, target, args, title):
        """
        Runs a function in a background thread, while showing a progress bar in the UI
        as a pop up window.
        """
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)

        parallel_event = threading.Event()
        dep_thread = threading.Thread(
            target=target,
            args=(
                *args,
                parallel_event,
            ),
        )
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()

    def cleanup(self):
        """
        Clean up resources when the module is closed.
        """
        self.removeObservers()

        if hasattr(self, "_qt_event_filters"):
            for slice_view, event_filter in self._qt_event_filters:
                slice_view.removeEventFilter(event_filter)
            self._qt_event_filters = []

        self.remove_shortcut_items()

    def __del__(self):
        """
        Called when the widget is destroyed.
        """
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()

        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(
                prompt_type["name"]
            ):
                debug_print("Skipping", prompt_name)
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()

            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)

            prompt_type["button"].setStyleSheet(
                f"""
                QPushButton {{
                    {self.unselected_style}
                }}
                QPushButton:checked {{
                    {self.selected_style}
                }}
            """
            )

            self.prev_caller = None

            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(
                    slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                    prompt_type["on_placed_function"],
                )

            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name)) 
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            # Note: Icons are not included in this implementation, but the code structure is preserved

        # To make sure that when segment is reset, no interaction is selected (without this code
        # the last interaction tool gets selected)
        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

    def is_ui_dark_or_light_mode(self):
        # Returns whether the current appearance of the UI is dark mode (will return "dark")
        # or light mode (will return "light")
        current_style = slicer.app.settings().value("Styles/Style")

        if current_style == "Dark Slicer":
            return "dark"
        elif current_style == "Light Slicer":
            return "light"
        elif current_style == "Slicer":
            app_palette = QApplication.instance().palette()
            window_color = app_palette.color(QPalette.Active, QPalette.Window)
            lightness = window_color.lightness()
            dark_mode_threshold = 128

            if lightness < dark_mode_threshold:
                return "dark"
            else:
                return "light"
        return "light"

    def remove_prompt_nodes(self):
        """
        Removes all the Markups/Fiducials prompts.
        """

        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    slicer.mrmlScene.RemoveNode(node)

        for prompt_type in list(self.prompt_types.values()):
            _remove(prompt_type["name"])

    def on_interaction_node_modified(self, caller, event):
        """
        Deselect prompt button if interaction mode is not place point anymore
        """

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        for prompt_type in self.prompt_types.values():
            if interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place:
                prompt_type["button"].setChecked(False)
            elif interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.Place:
                placingThisNode = (selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID())
                prompt_type["button"].setChecked(placingThisNode)

    def remove_all_but_last_prompt(self):
        """
        Removes all but the most recently placed markup points
        (helpful when segment change was detected).
        """
        last_modified_node = None
        all_nodes = []

        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes and existing_nodes.GetNumberOfItems() > 0:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)

                    all_nodes.append(node)
                    if (
                        last_modified_node is None
                        or node.GetMTime() > last_modified_node.GetMTime()
                    ):
                        last_modified_node = node

        for node in all_nodes:
            n = node.GetNumberOfControlPoints()

            if node == last_modified_node:
                n -= 1

            for i in range(n):
                node.RemoveNthControlPoint(0)

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if checked:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(self.prompt_types[prompt_name]["node_class"])
            selectionNode.SetActivePlaceNodeID(self.prompt_types[prompt_name]["node"].GetID())
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def display_node_markup_point(self, display_node):
        """
        Handles the appearance of the point display node.
        """
        display_node.SetTextScale(0)  # Hide text labels
        display_node.SetGlyphScale(0.75)  # Make the points larger
        display_node.SetColor(0.0, 0.0, 1.0)  # Blue color
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)  # Fully opaque
        display_node.SetSliceProjection(False)  # Make points visible in all slice views

    def display_node_markup_bbox(self, display_node):
        """
        Handles the appearance of the BBox display node.
        """
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetSliceProjectionColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    ###############################################################################
    # Event handlers for prompts
    ###############################################################################

    #
    #  -- Point
    #
    def on_point_placed(self, caller, event):
        """
        Called when a point is placed in the scene. Grabs the point position
        and sends it to the server.
        """
        xyz = self.xyz_from_caller(caller)

        volume_node = self.get_volume_node()
        if volume_node:
            self.point_prompt(xyz=xyz, positive_click=self.is_positive)

    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        """
        Uploads point prompt to the server.
        """
        url = f"{self.server}/add_point_interaction"

        seg_response = self.request_to_server(
            url, json={"voxel_coord": xyz[::-1], "positive_click": positive_click}
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        debug_print("unpacked_segmentation.sum():", unpacked_segmentation.sum())
        debug_print(seg_response)
        debug_print(f"{positive_click} point prompt triggered! {xyz}")

        self.show_segmentation(unpacked_segmentation)

    #
    #  -- Bounding Box
    #
    def on_bbox_placed(self, caller, event):
        """
        Every time a control point is placed/moved for the bounding box ROI node.
        Once two corners are placed, we send the bounding box to the server.
        """
        xyz = self.xyz_from_caller(caller)

        if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
            roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0
            roi_node.SetSize(current_size)

            volume_node = self.get_volume_node()
            if volume_node:
                outer_point_two = self.prev_bbox_xyz

                outer_point_one = [
                    xyz[0] * 2 - outer_point_two[0],
                    xyz[1] * 2 - outer_point_two[1],
                    xyz[2] * 2 - outer_point_two[2],
                ]

                self.bbox_prompt(
                    outer_point_one=outer_point_one,
                    outer_point_two=outer_point_two,
                    positive_click=self.is_positive,
                )

                def _next():
                    self.setup_prompts()
                    # Start placing a new box
                    self.ui.pbInteractionBBox.click()

                qt.QTimer.singleShot(0, _next)

            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz

        self.prev_caller = caller

    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        """
        Uploads BBox prompt to the server.
        """
        url = f"{self.server}/add_bbox_interaction"

        seg_response = self.request_to_server(
            url,
            json={
                "outer_point_one": outer_point_one[::-1],
                "outer_point_two": outer_point_two[::-1],
                "positive_click": positive_click,
            },
        )

        unpacked_segmentation = self.unpack_binary_segmentation(
            seg_response.content, decompress=False
        )
        self.show_segmentation(unpacked_segmentation)

    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def make_new_segment(self):
        """
        Creates a new empty segment in the current segmentation, increments a name,
        and sets it as the selected segment.
        """
        # After creating a new segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        debug_print("doing make_new_segment")
        segmentation_node = self.get_segmentation_node()

        # Generate a new segment name
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0:
            new_segment_name = "Segment_1"
        else:
            # Find the next available number
            segment_numbers = [
                int(seg.split("_")[-1])
                for seg in segment_ids
                if seg.startswith("Segment_") and seg.split("_")[-1].isdigit()
            ]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"

        # Create and add the new segment
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(
            new_segment_name
        )
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        # Make sure the right node is selected
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)

        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        """
        Clears the contents (labelmap) of the currently selected segment
        and updates the server.
        """
        # After clearing a segment, negative prompts do not make sense, so
        # we're automatically switching the prompt type to positive.
        self.ui.pbPromptTypePositive.click()
        
        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()

        if selected_segment_id:
            debug_print(f"Clearing segment: {selected_segment_id}")
            self.show_segmentation(
                np.zeros(self.get_image_data().shape, dtype=np.uint8)
            )
            self.setup_prompts()
            self.upload_segment_to_server()
        else:
            debug_print("No segment selected to clear.")

    def show_segmentation(self, segmentation_mask):
        """
        Updates the currently selected segment with the given binary mask array.
        Handles both 2D slice segmentation and 3D volume segmentation.
        """
        t0 = time.time()
        
        # Get segmentation node and segment ID
        segmentationNode, selectedSegmentID = (
            self.get_selected_segmentation_node_and_segment_id()
        )
        volumeNode = self.get_volume_node()
        if not volumeNode:
            debug_print("No volume node found")
            return
        
        # Check if segmentation_mask is 2D (from server) or 3D (from other sources)
        if segmentation_mask.ndim == 2:
            # Handle 2D segmentation result (from server)
            
            # Get current slice index
            sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
            if not sliceWidget:
                debug_print("No active slice widget found")
                return
            
            sliceLogic = sliceWidget.sliceLogic()
            sliceNode = sliceLogic.GetSliceNode()
            
            # Get slice offset
            sliceOffset = sliceNode.GetSliceOffset()
            
            # Get volume image data
            imageData = volumeNode.GetImageData()
            if not imageData:
                debug_print("No image data found in volume node")
                return
            
            # Calculate slice index from offset
            spacing = imageData.GetSpacing()
            origin = imageData.GetOrigin()
            sliceIndex = int(round((sliceOffset - origin[2]) / spacing[2]))
            
            # Get current 3D segmentation data
            currentSegmentation = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentationNode, selectedSegmentID, volumeNode
            )
            
            # Update the specific slice with the 2D segmentation result
            if sliceIndex >= 0 and sliceIndex < currentSegmentation.shape[0]:
                currentSegmentation[sliceIndex, :, :] = segmentation_mask
                
                # Save updated segmentation to previous_states
                self.previous_states["segment_data"] = currentSegmentation
                
                # Update the segment with the full 3D segmentation data
                with slicer.util.RenderBlocker():  # avoid flashing of 3D view
                    self.ui.editor_widget.saveStateForUndo()
                    slicer.util.updateSegmentBinaryLabelmapFromArray(
                        currentSegmentation,
                        segmentationNode,
                        selectedSegmentID,
                        volumeNode,
                    )
        else:
            # Handle 3D segmentation result (original behavior)
            self.previous_states["segment_data"] = segmentation_mask
            
            with slicer.util.RenderBlocker():
                self.ui.editor_widget.saveStateForUndo()
                slicer.util.updateSegmentBinaryLabelmapFromArray(
                    segmentation_mask,
                    segmentationNode,
                    selectedSegmentID,
                    volumeNode,
                )
        
        # Handle 3D representation if needed
        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())
        if was_3d_shown:
            segmentationNode.CreateClosedSurfaceRepresentation()

        # Mark the segment as being edited
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)

        # Mark the segmentation as modified so the UI updates
        segmentationNode.Modified()

        if segmentation_mask.sum() > 0:
            # If we do this when segmentation_mask.sum() == 0, sometimes Slicer will throw "bogus" OOM errors
            # (see https://github.com/coendevente/SlicerNNInteractive/issues/38)
            segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()
        
        del segmentation_mask

        debug_print(f"show_segmentation took {time.time() - t0}")

    def get_segmentation_node(self):
        """
        Returns the currently referenced segmentation node (from the Segment Editor).
        If none exists, we create a fresh one.
        """
        # If the segmentation widget has a currently selected segmentation node, return it.
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node:
            return segmentation_node

        # Otherwise, fall back to getting the first suitable segmentation node
        segmentation_node = None
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for segmentation_node in segmentation_nodes:
            pass  # Just get the first one

        # Create new segmentation node if none suitable found
        if not segmentation_node:
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Set segmentation node in widget
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())

        return segmentation_node

    def get_selected_segmentation_node_and_segment_id(self):
        """
        Retrieve the currently selected segmentation node & segment ID.
        If none, create one.
        """
        debug_print("doing get_selected_segmentation_node_and_segment_id")
        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id:
            return self.make_new_segment()

        return segmentation_node, selected_segment_id

    def get_current_segment_id(self):
        """
        Returns the ID of the segment currently selected in the segment editor.
        """
        return self.ui.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

    def get_segment_data(self):
        """
        Gets the labelmap array (binary) of the currently selected segment.
        """
        segmentation_node, selected_segment_id = (
            self.get_selected_segmentation_node_and_segment_id()
        )

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, selected_segment_id, self.get_volume_node()
        )
        seg_data_bool = mask.astype(bool)

        return seg_data_bool

    def selected_segment_changed(self):
        """
        Checks if the current segment mask has changed from our `self.previous_states`.
        """
        segment_data = self.get_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        selected_segment_changed = old_segment_data is None or not np.array_equal(
            old_segment_data.astype(bool), segment_data.astype(bool)
        )

        debug_print(f"segment_data.sum(): {segment_data.sum()}")

        if old_segment_data is not None:
            debug_print(f"old_segment_data.sum(): {old_segment_data.sum()}")
        else:
            debug_print("old_segment_data is None")

        debug_print(f"selected_segment_changed: {selected_segment_changed}")

        return selected_segment_changed

    ###############################################################################
    # Server communication and sync functions
    ###############################################################################

    def update_server(self):
        """
        Reads user-entered server URL from UI, saves to QSettings, updates self.server.
        """
        self.server = self.ui.Server.text.rstrip("/")
        settings = qt.QSettings()
        settings.setValue("MedSAM2/server", self.server)
        debug_print(f"Server URL updated and saved: {self.server}")

    def test_server_connection(self):
        """
        Sends a lightweight GET request to see if the configured server responds.
        """
        server_text = self.ui.Server.text
        if not server_text.strip():
            QMessageBox.warning(
                slicer.util.mainWindow(),
                "Test Connection",
                "Please enter a server URL before testing the connection.",
            )
            return

        self.ui.Server.setText(server_text.strip())
        self.update_server()
        server_url = self.server

        if getattr(self, "_test_server_in_progress", False):
            return
        self._test_server_in_progress = True

        slicer.util.showStatusMessage("Testing MedSAM2 server connection...", 2000)

        def test_connection():
            try:
                url = f"{server_url}/ping"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    QMessageBox.information(
                        slicer.util.mainWindow(),
                        "Test Connection",
                        "Server connection successful!",
                    )
                else:
                    QMessageBox.warning(
                        slicer.util.mainWindow(),
                        "Test Connection",
                        f"Server returned unexpected status code: {response.status_code}",
                    )
            except requests.exceptions.RequestException as e:
                QMessageBox.critical(
                    slicer.util.mainWindow(),
                    "Test Connection",
                    f"Failed to connect to server: {str(e)}",
                )
            finally:
                self._test_server_in_progress = False

        import threading
        test_thread = threading.Thread(target=test_connection)
        test_thread.start()

    def request_to_server(self, url, json=None, data=None, headers=None):
        """
        Sends a request to the server and returns the response.
        """
        if headers is None:
            headers = {}

        try:
            response = requests.post(url, json=json, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(
                slicer.util.mainWindow(),
                "Server Error",
                f"Failed to communicate with server: {str(e)}",
            )
            return None

    def upload_image_to_server(self):
        """
        Uploads the current 2D slice image to the server with window/level applied.
        """
        url = f"{self.server}/upload_image"
        
        # Get current slice image data with window/level applied
        image_data = self.get_current_slice_data_with_window_level()
        if image_data is None:
            return None

        # Convert image data to bytes (2D numpy array)
        buffer = io.BytesIO()
        np.save(buffer, image_data)
        compressed_data = gzip.compress(buffer.getvalue())

        # Send to server
        from requests_toolbelt import MultipartEncoder
        fields = {
            "file": ("slice.npy.gz", compressed_data, "application/octet-stream"),
        }
        encoder = MultipartEncoder(fields=fields)
        response = self.request_to_server(
            url,
            data=encoder,
            headers={"Content-Type": encoder.content_type},
        )

        if response is not None:
            self.previous_states["image_data"] = image_data
            debug_print("2D slice image uploaded successfully!")
        return response
        
    def get_current_slice_data_with_window_level(self):
        """
        Gets the current slice image data with window/level applied.
        Returns a 2D numpy array.
        """
        # Get current active slice view
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        if not sliceWidget:
            debug_print("No active slice widget found")
            return None
        
        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        
        # Get current slice index
        volumeNode = self.get_volume_node()
        if not volumeNode:
            debug_print("No volume node found")
            return None
        
        # Get slice offset
        sliceOffset = sliceNode.GetSliceOffset()
        
        # Get volume image data
        imageData = volumeNode.GetImageData()
        if not imageData:
            debug_print("No image data found in volume node")
            return None
        
        # Get volume dimensions and spacing
        extent = imageData.GetExtent()
        spacing = imageData.GetSpacing()
        origin = imageData.GetOrigin()
        
        # Calculate slice index from offset
        sliceIndex = int(round((sliceOffset - origin[2]) / spacing[2]))
        
        # Get 3D volume array
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        
        # Extract 2D slice (assuming volumeArray is in format [Z, Y, X])
        if sliceIndex < 0 or sliceIndex >= volumeArray.shape[0]:
            debug_print(f"Invalid slice index: {sliceIndex}")
            return None
        
        sliceArray = volumeArray[sliceIndex, :, :]
        
        # Get current window/level settings
        windowWidth = sliceNode.GetWindowWidth()
        windowCenter = sliceNode.GetWindowCenter()
        
        # Apply window/level to slice
        sliceWithWindowLevel = self.apply_window_level(sliceArray, windowCenter, windowWidth)
        
        # Convert to MedSAM2 expected format (0-255 range, 3-channel if needed)
        sliceWithWindowLevel = self.convert_to_medsam2_format(sliceWithWindowLevel)
        
        return sliceWithWindowLevel
        
    def apply_window_level(self, imageArray, windowCenter, windowWidth):
        """
        Applies window/level to the image array.
        
        Args:
            imageArray: 2D numpy array of image data
            windowCenter: Window center value
            windowWidth: Window width value
            
        Returns:
            2D numpy array with window/level applied
        """
        # Calculate window min and max
        windowMin = windowCenter - windowWidth / 2
        windowMax = windowCenter + windowWidth / 2
        
        # Apply window/level
        result = np.clip(imageArray, windowMin, windowMax)
        result = (result - windowMin) / (windowMax - windowMin)  # Normalize to 0-1
        
        return result
        
    def convert_to_medsam2_format(self, imageArray):
        """
        Converts the image array to MedSAM2 expected format.
        
        Args:
            imageArray: 2D numpy array with values in 0-1 range
            
        Returns:
            2D numpy array in MedSAM2 format
        """
        # MedSAM2 expects 0-255 range, uint8
        imageArray = (imageArray * 255).astype(np.uint8)
        
        # If MedSAM2 expects 3-channel input, uncomment the following line
        # imageArray = np.stack((imageArray, imageArray, imageArray), axis=-1)
        
        return imageArray

    def upload_segment_to_server(self):
        """
        Uploads the current 2D slice segmentation to the server.
        """
        url = f"{self.server}/upload_segment"
        
        # Get current 2D slice segmentation data
        segment_data = self.get_current_slice_segment_data()
        if segment_data is None:
            return None

        # Convert segment data to bytes
        buffer = io.BytesIO()
        np.save(buffer, segment_data.astype(np.uint8))
        compressed_data = gzip.compress(buffer.getvalue())

        # Send to server
        from requests_toolbelt import MultipartEncoder
        fields = {
            "file": ("segment.npy.gz", compressed_data, "application/octet-stream"),
        }
        encoder = MultipartEncoder(fields=fields)
        response = self.request_to_server(
            url,
            data=encoder,
            headers={"Content-Type": encoder.content_type},
        )

        if response is not None:
            self.previous_states["segment_data"] = segment_data
            debug_print("2D segment uploaded successfully!")
        return response
        
    def get_current_slice_segment_data(self):
        """
        Gets the current slice segmentation data.
        Returns a 2D numpy array.
        """
        # Get current slice index
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        if not sliceWidget:
            debug_print("No active slice widget found")
            return None
        
        sliceLogic = sliceWidget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        
        volumeNode = self.get_volume_node()
        if not volumeNode:
            debug_print("No volume node found")
            return None
        
        # Get slice offset
        sliceOffset = sliceNode.GetSliceOffset()
        
        # Get volume image data
        imageData = volumeNode.GetImageData()
        if not imageData:
            debug_print("No image data found in volume node")
            return None
        
        # Calculate slice index from offset
        spacing = imageData.GetSpacing()
        origin = imageData.GetOrigin()
        sliceIndex = int(round((sliceOffset - origin[2]) / spacing[2]))
        
        # Get segmentation data
        segmentation_node, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()
        if not segmentation_node or not selected_segment_id:
            debug_print("No segmentation node or segment ID found")
            return None
        
        # Get 3D segment data
        segment_3d = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, selected_segment_id, volumeNode
        )
        
        # Extract 2D slice (assuming segment_3d is in format [Z, Y, X])
        if sliceIndex < 0 or sliceIndex >= segment_3d.shape[0]:
            debug_print(f"Invalid slice index: {sliceIndex}")
            return None
        
        segment_2d = segment_3d[sliceIndex, :, :]
        
        return segment_2d

    def unpack_binary_segmentation(self, content, decompress=True):
        """
        Unpacks the binary segmentation received from the server.
        """
        if decompress:
            content = gzip.decompress(content)
        buffer = io.BytesIO(content)
        return np.load(buffer)

    def image_changed(self):
        """
        Checks if the current image has changed from our `self.previous_states`.
        """
        image_data = self.get_image_data()
        old_image_data = self.previous_states.get("image_data", None)
        image_changed = old_image_data is None or not np.array_equal(
            old_image_data, image_data
        )
        return image_changed

    def get_image_data(self):
        """
        Gets the image data from the current volume node.
        """
        volume_node = self.get_volume_node()
        if not volume_node:
            return None

        image_data = slicer.util.arrayFromVolume(volume_node)
        return image_data

    def get_volume_node(self):
        """
        Gets the current volume node from the segment editor.
        """
        return self.ui.editor_widget.sourceVolumeNode()

    def xyz_from_caller(self, caller, point_type="control_point"):
        """
        Extracts xyz coordinates from the caller node.
        """
        if point_type == "control_point":
            # Get the last placed point
            point_id = caller.GetNumberOfControlPoints() - 1
            if point_id < 0:
                return []
            xyz = [0.0, 0.0, 0.0]
            caller.GetNthControlPointPosition(point_id, xyz)
            return xyz
        elif point_type == "curve_point":
            # Get all curve points
            xyzs = []
            for i in range(caller.GetNumberOfControlPoints()):
                xyz = [0.0, 0.0, 0.0]
                caller.GetNthControlPointPosition(i, xyz)
                xyzs.append(xyz)
            return xyzs

    @property
    def is_positive(self):
        """
        Returns True if the current prompt type is positive.
        """
        return self.current_prompt_type_positive

    def on_prompt_type_positive_clicked(self):
        """
        Called when the positive prompt type button is clicked.
        """
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

    def on_prompt_type_negative_clicked(self):
        """
        Called when the negative prompt type button is clicked.
        """
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)

    def toggle_prompt_type(self):
        """
        Toggles between positive and negative prompt types.
        """
        if self.current_prompt_type_positive:
            self.ui.pbPromptTypeNegative.click()
        else:
            self.ui.pbPromptTypePositive.click()


###############################################################################
# MedSAM2Logic
###############################################################################


class MedSAM2Logic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)


###############################################################################
# MedSAM2Test
###############################################################################


class MedSAM2Test(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MedSAM21()

    def test_MedSAM21(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Test basic module setup
        module = slicer.modules.MedSAM2
        self.assertIsNotNone(module, "MedSAM2 module is None")
        self.assertEqual(module.title, "MedSAM2", "Module title is incorrect")

        self.delayDisplay("Test passed")